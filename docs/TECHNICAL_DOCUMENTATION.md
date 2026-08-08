# RAG Document Q&A Chatbot - Technical Documentation

Full implementation reference, written from the actual codebase. Intended for a developer picking up or reviewing this project.

## 1. System Overview

A Retrieval-Augmented Generation chatbot: users upload documents, the system chunks and embeds them into Pinecone, and questions are answered by retrieving relevant chunks and streaming an LLM-generated answer grounded in those chunks, with inline source citations (file + page).

**Production stack:**

```
Frontend (vanilla HTML/CSS/JS)
        │  fetch() — SSE stream
        ▼
FastAPI backend (main.py)
        │
        ├── pinecone_store.py → Pinecone (vector DB)
        ├── LlamaIndex Settings → Groq LLaMA 3.3 70B (LLM)
        └──  → HuggingFace MiniLM-L6-v2 (embeddings)
```

**Git Repo:** https://github.com/bansalmannatbansal/RAG_Pinecone

**Repo files:**

| File | Role |
|---|---|
| `main.py` | Production backend. FastAPI app, all routes, retrieval/prompt/streaming logic. |
| `app.py` | Legacy prototype. Streamlit version of the same logic. Superseded by `main.py`; kept for reference only — not run in production. |
| `pinecone_store.py` | Vector store layer: index creation, PDF loading (PyMuPDF + Docling/EasyOCR fallback), text cleaning, build/load/clear index. |
| `index.html` / `app.js` / `style.css` | Frontend: upload UI, chat UI, SSE consumer. |
| `debug_pdf.py` | Standalone CLI script to sanity-check that PyMuPDF extraction works before indexing. |
| `diagnose.py` | Standalone CLI script that pinpoints which text-cleaning guard is rejecting a given PDF's pages. |
| `requirements.txt` | Dependencies. |

## 2. Environment & Setup

- **Python:** 3.14, macOS, local venv at `./venv/`
- **Run command:** `python -m uvicorn main:app --reload --port 8000`
- **Required env vars** (loaded via `python-dotenv` from `.env`):
  - `GROQ_API_KEY`
  - `PINECONE_API_KEY`
  - `PINECONE_INDEX` (optional, defaults to `"rag-documents"`)
- **Expected file layout** (referenced directly in code):
  - `main.py`
  - `pinecone_store.py`
  - `templates/index.html` ← main.py reads this path directly
  - `static/app.js`
  - `static/style.css`
- **SSL fix:** `pinecone_store.py` patches the default SSL context using `certifi` before loading Docling — a workaround needed on macOS Python 3.14 where HuggingFace/Docling model downloads otherwise fail SSL verification.
- **Install:** `pip install -r requirements.txt` (use `pip3` on Mac per local convention). OCR dependencies (`pytesseract`, `pdf2image`) are commented out in `requirements.txt` — the actual OCR fallback used is Docling + EasyOCR, not pytesseract (see Section 4.2).

## 3. Configuration Reference

All config is hardcoded as module-level constants in `main.py` (mirrored in `app.py` for the legacy version) — none of it is user-exposed.

| Constant | Value | Effect |
|---|---|---|
| `MODEL_NAME` | `"llama-3.3-70b-versatile"` | Groq model used for both answering and query rewriting |
| `TOP_K` | 25 | Chunks pulled per retrieval call |
| `CHUNK_SIZE` | 512 | LlamaIndex chunking size (tokens) at index time |
| `CHUNK_OVERLAP` | 50 | Overlap between adjacent chunks |
| `MAX_FILE_SIZE` | 20 MB | Per-file upload limit |
| `ALLOWED_TYPES` | pdf, txt, md, docx, csv, xlsx, json, pptx | Accepted upload extensions |
| `MIN_SCORE` | 0.20 | Similarity-score floor for a retrieved chunk to be used |
| `MAX_CHUNKS_PER_FILE` | 5 | Caps how many chunks from one file enter the prompt |
| `MAX_CHUNK_CHARS` | 6000 | Total context character budget before truncation |
| `MAX_HISTORY_CHARS` | 600 | Cap on injected recent-question history text |
| `MAX_NON_LATIN` | 0.25 | Max allowed ratio of non-Latin characters in a chunk before it's discarded as corrupt |
| `MAX_PROMPT_CHARS` | 11000 | Hard cap on the full assembled prompt sent to Groq |

## 4. `pinecone_store.py` — Vector Store Layer

### 4.1 Pinecone Index Management

`_get_pinecone_index()` connects via the pinecone client, lists existing indexes, and creates one if missing:

```python
pc.create_index(
    name=PINECONE_INDEX,
    dimension=384,  # matches MiniLM-L6-v2 output dim
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
```

`dimension=384` is hardcoded to match the embedding model; changing the embedding model requires updating this value and rebuilding the index from scratch (existing vectors are dimension-locked).

### 4.2 PDF Loading Strategy — Two-Tier Fallback

This is the most involved part of the codebase. `_load_pdf_docling(fpath)` implements a try-fast-first, fall-back-to-OCR strategy:

1. **Tier 1 — PyMuPDF** (`_load_pdf_pymupdf`): opens the PDF with `pymupdf.open()`, calls `.get_text()` per page. Fast, no OCR overhead. Returns one `Document` per page that has non-empty extracted text, with metadata `file_name`, `file_path`, `page_label`, `page_number`.
2. **Tier 2 — Docling + EasyOCR:** only triggered if Tier 1 returns zero documents (i.e., the PDF is scanned/image-based with no embedded text layer). Configures `PdfPipelineOptions` with `do_ocr=True` and `EasyOcrOptions(force_full_page_ocr=True)`, then runs `DocumentConverter.convert()` and exports to markdown via `doc_obj.export_to_markdown()`.

Page-level splitting of Docling output (`_split_into_pages`) first attempts to use Docling's internal item provenance (`item.prov[].page_no`) to group text by actual page number. If that structure isn't available, it falls back to naive 2000-character chunking with sequential fake "page" numbers — meaning OCR'd documents may have less accurate page citations than native-text PDFs.

Non-PDF files (`.txt`, `.md`, `.docx`, `.csv`, `.xlsx`, `.json`, `.pptx`) go through LlamaIndex's `SimpleDirectoryReader` instead, with `filename_as_id=True`.

### 4.3 Text Cleaning Pipeline

`_clean_doc_text(text)` runs on every loaded chunk before it's allowed into the index, in this order:

1. Strip `|W|` markers (web-scrape export artifacts) and `[Source N — filename]` annotation lines via regex.
2. Force UTF-8 re-encode with `errors="ignore"` to drop invalid byte sequences.
3. Replace any Unicode control character (`unicodedata.category(ch) == "Cc"`) except tab/newline with a space.
4. Collapse repeated spaces, strip whitespace.
5. **Non-Latin ratio guard:** if `len(text) > 0`, compute the fraction of characters with `ord(ch) > 0x024F` (i.e., outside Latin/Latin Extended-A/B ranges). If that ratio exceeds `MAX_NON_LATIN` (0.25), the chunk is discarded — this catches corrupted PDF font-encoding output (pages that decode as garbled Arabic/Kurdish-looking characters from a broken font map).

If a chunk fails any guard, `_clean_doc_text` returns `None` and `build_index` increments a skipped counter rather than indexing it.

### 4.4 `build_index()` Flow

1. Get/create Pinecone index.
2. `pinecone_index.delete(delete_all=True)` — wrapped in try/except: throws on empty namespace, which is caught and logged rather than raised.
3. For each file in `TMP_DIR`:
   - `.pdf` → `_load_pdf_docling()` (PyMuPDF → Docling/EasyOCR fallback)
   - other → `SimpleDirectoryReader`
   - any raw doc starting with `"%PDF"` (extraction failure leaking raw bytes) → skipped
   - `_clean_doc_text()` on every doc → skipped if `None`
4. If `clean_docs` is empty → raise `ValueError` listing failed files.
5. `VectorStoreIndex.from_documents(clean_docs, storage_context=...)` → this is what actually embeds + upserts into Pinecone.

### 4.5 `load_index()` and Utilities

`load_index()` checks `describe_index_stats()["total_vector_count"]`; if zero, returns `None` (so the app correctly shows an empty state on startup rather than a broken retriever). Otherwise reconstructs a `VectorStoreIndex` from the existing Pinecone vector store — this is what allows the app to survive restarts without re-uploading documents.

`get_chunk_count()` and `clear_index()` are thin wrappers, both defensively wrapped in try/except around the Pinecone calls.

## 5. `main.py` — FastAPI Backend

### 5.1 App Setup & State

- CORS is wide open (`allow_origins=["*"]`) — fine for local/internal use, would need tightening for any public deployment.
- `/static` is mounted from a `static/` directory for `app.js`/`style.css`.
- State is a single in-memory dict, not a database or session store:

```python
state = {
    "retriever": None,
    "indexed_files": [],
    "chat_history": [],
    "chunk_count": 0,
}
```

This means: single-process only, no multi-user isolation, and all chat history/indexed-file tracking is lost on server restart (though the Pinecone index itself persists — `load_index()` on startup reconnects the retriever, but `indexed_files` and `chat_history` do not repopulate from anywhere, since they're not stored in Pinecone metadata at the app level).

- `init_settings()` configures the global LlamaIndex `Settings` singleton (LLM = Groq, embed model = MiniLM) exactly once, guarded by checking `Settings._llm is None` — this avoids LlamaIndex's default OpenAI-resolution behavior firing before the Groq LLM is set.
- `@app.on_event("startup")` calls `init_settings()` and `load_index()`, so an existing Pinecone index is usable immediately without re-building.

### 5.2 Routes

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Serves `templates/index.html` directly as `HTMLResponse` |
| GET | `/status` | Returns `indexed_files`, `chunk_count`, `ready` (whether a retriever is loaded) |
| POST | `/upload` | Multipart file upload; validates extension + 20MB size limit; saves to `TMP_DIR` (`/tmp/rag_docs`) |
| POST | `/build` | Calls `build_index()`, rebuilds the retriever, refreshes `indexed_files` from disk, clears chat history |
| POST | `/clear` | Clears Pinecone index, resets all in-memory state, deletes `TMP_DIR` |
| POST | `/chat` | Main Q&A endpoint — returns an SSE `StreamingResponse` (see 5.4) |
| GET | `/history` | Returns current in-memory chat history |
| POST | `/clear-history` | Clears chat history only (keeps the index) |

Note: `/upload` and `/build` are separate steps — files are saved to disk first, then indexed in a second explicit call. The frontend calls them back-to-back (Section 6).

### 5.3 Query Rewriting (`build_standalone_query`)

Before retrieval, follow-up questions are optionally rewritten into standalone queries:

- Only triggers if the current question contains a pronoun/anaphora word from a fixed set (`it`, `that`, `this`, `they`, `here`, `made`, `built`, etc.) — a cheap heuristic to avoid burning an extra LLM call on every message.
- If triggered, sends a small prompt to Groq with the last user question + a 200-char slice of the last assistant answer, asking for a rewritten standalone query.
- Defensive: if the rewritten result is empty or implausibly long (>300 chars), or the call throws, it silently falls back to the original `user_input`.

### 5.4 Cross-Document Retrieval

Two cooperating functions:

- `is_cross_document_query(user_input, indexed_files)` — returns `True` if (a) the question contains an explicit comparison word (`both`, `each`, `across`, `compare`, `versus`, `vs`), or (b) keyword-matching (via `_file_keywords`) finds tokens from 2 or more different indexed filenames present in the question text. Always `False` if fewer than 2 files are indexed.
- `_file_keywords(fname)` — derives meaningful search tokens from a filename: splits on underscores/spaces, splits camelCase boundaries, drops stopwords, and additionally tries to decompose long concatenated tokens (>8 chars) using a small hardcoded `KNOWN_WORDS` list (e.g., `"thelinuxcommandline"` → `["linux", "command", "line"]`). This `KNOWN_WORDS` list is project/dataset-specific (contains words like `sahayak`, `backlink`, `capstone` from the actual test documents used) — it will need extending for new document sets to keep cross-doc detection reliable.
- `retrieve_cross_document(user_input)` — when cross-doc is detected, instead of a single embedding query, runs one sub-query per indexed file (`f"{user_input} — focus on {stem}"`), retrieves against each, deduplicates by node `id()`, and merges + re-sorts by score. This directly implements the "Cross-Document Retrieval Tuning" decision: a single query embedding sitting semantically between two document spaces would otherwise score too low against either to clear `MIN_SCORE`.

### 5.5 The `/chat` Endpoint — Full Pipeline

This is the core of the application. Inside the `generate()` async generator:

**Step 1 — History & query rewrite.** Append user message to `state["chat_history"]`, call `build_standalone_query`.

**Step 2 — Retrieval.** Branches on `is_cross_document_query`: either `retrieve_cross_document()` or a direct `state["retriever"].retrieve(standalone)`. Wrapped in try/except — on failure, yields an error SSE event and returns immediately.

**Step 3 — Score filtering + cleaning.** Keep nodes with `score >= MIN_SCORE`. For each, run `strip_markers()` then `sanitise_text()` on content; nodes that clean to nothing are dropped. Fallback: if zero nodes survive at `MIN_SCORE`, retry once at `max(0.10, MIN_SCORE - 0.05)` — a soft floor so a near-miss question doesn't return "nothing found" when reasonably relevant content exists just below the main threshold.

**Step 4 — No-results path.** If still nothing, stream a fixed "couldn't find sufficiently relevant information" message and end the generator early (still emits a `done` event with empty sources, so the frontend always gets a terminal event).

**Step 5 — Context assembly with three-layer deduplication:**

- **Fingerprint dedup:** skip a chunk if its first 60 characters match one already included (catches near-duplicate retrievals, e.g. overlapping chunks from `CHUNK_OVERLAP`).
- **Per-file cap:** stop adding chunks from a file once `MAX_CHUNKS_PER_FILE` (5) is reached.
- **Char budget:** stop entirely once `total_chars` would exceed `MAX_CHUNK_CHARS` (6000).
- **Sentence-level dedup pass:** after assembly, a second pass walks each context part line-by-line; any line ≥20 chars that has already been seen verbatim (across any chunk) is dropped, while keeping the `[Source — ...]` header line. This catches repeated sentences across overlapping chunks that the chunk-level fingerprint dedup wouldn't catch.

**Step 6 — Prompt construction.** Fixed system prompt instructs the model to answer only from provided sources, never use outside knowledge, never repeat itself, and return an exact fallback string (`"This information is not found in the uploaded documents."`) when the answer isn't present. History (last 2 user questions, capped at `MAX_HISTORY_CHARS`) is prepended. If the full assembled prompt exceeds `MAX_PROMPT_CHARS` (11000), the context section is truncated by the overflow amount and the prompt rebuilt — this is the concrete mechanism behind the "prompt hard cap" decision, preventing Groq 413 errors.

**Step 7 — Streaming.** Iterates `Settings.llm.stream_complete(full_prompt)`, reading `getattr(token, "delta", None)` per token. Any falsy or non-string delta is skipped (this is the production equivalent of the manual-accumulator fix developed against the Streamlit `st.write_stream` "None" badge bug — here it's a generator-level guard instead of a UI-level workaround). Each valid delta is both accumulated into `answer_text` and yielded immediately as an SSE `text` event.

**Step 8 — Sources.** Only computed if the answer doesn't contain the literal "not found in the uploaded documents" fallback phrase, via `format_sources()`.

**Step 9 — Final event + history.** Appends the full answer to `state["chat_history"]` (capped at 40 messages, oldest trimmed), then yields a terminal SSE `done` event carrying the sources array.

### 5.6 `format_sources()` — Citation Logic

Three guards, applied in order:

1. **Ghost-citation guard:** drop any node whose `file_name` is missing/empty/the string `"None"`, or — if `indexed_files` is non-empty — whose filename isn't actually in the currently indexed set (prevents citing a file from a stale/previous index).
2. **Relevance proximity filter:** drop any source whose score is more than 0.10 below the top score among the surviving nodes — keeps citations to the genuinely best-matching chunks rather than padding the list with marginal ones.
3. **Dedup by (filename, page):** if the same file+page appears from multiple chunks, keep only the highest-scoring instance.

Result is sorted by score descending, capped at the top 5, and returned as a list of `{file, page, score}` dicts — page is explicitly stringified-or-None (never the raw string `"None"`), which is the production fix for the metadata-null edge case.

### 5.7 SSE Wire Format

Every event is a line of the form: `data: {"type": "text" | "done" | "error", ...}\n\n`

- `text` events carry incremental content (a delta string) — frontend appends and re-renders.
- `done` carries `sources` (list of citation dicts) — terminal event for a turn.
- `error` carries `content` (an error string) — also effectively terminal from the frontend's perspective.

## 6. Frontend (`index.html`, `app.js`, `style.css`)

### 6.1 Structure

Two-pane layout via CSS grid: a 280px sidebar (upload + status + indexed files) and a flex chat column (header + scrollable messages + input row). Dark theme defined entirely through CSS custom properties in `:root` (`--bg`, `--accent`, `--surface`, etc.) — changing the theme is a matter of editing those variables, not hunting through rules.

### 6.2 Upload → Build Flow (`buildIndex()` in `app.js`)

1. Client-side validation in `addFiles()`: extension allow-list and 20MB size check, mirroring (but not replacing) the server-side checks in `/upload`.
2. `POST /upload` with `FormData` containing all pending files.
3. If upload succeeds with at least one saved file, `POST /build` is called next — this is a two-step network round trip, with the button label updating between "Uploading..." and "Building index..." to reflect which phase is active.
4. On success: clears `pendingFiles`, re-renders the (now empty) file list, updates the status badge, and shows a success toast with the resulting chunk count.

### 6.3 Chat Flow (`sendMessage()` in `app.js`)

1. Appends the user bubble immediately, then appends an assistant bubble showing a CSS-animated typing indicator (three bouncing dots).
2. `POST /chat`, then reads the response body via `res.body.getReader()` — a manual streaming reader rather than `EventSource`, which is necessary here because `EventSource` only supports GET requests and this endpoint is a POST.
3. Buffers raw bytes, splits on `\n`, and processes only lines prefixed `data: `, JSON-parsing the remainder. Incomplete trailing lines are kept in `buffer` for the next chunk (standard SSE-over-fetch buffering pattern).
4. On `type: "text"` — appends to a running `fullText` and calls `updateBubbleText()`, which re-renders the bubble's HTML via `formatText()` on every token (note: this means each token triggers a full `innerHTML` rewrite of the bubble — fine at this conversation scale, but a potential perf consideration for very long answers).
5. On `type: "done"` — calls `updateBubble()` with the final text + sources, which appends a formatted "📎 Sources" block underneath the answer.
6. On `type: "error"` — replaces the bubble content with a ⚠️-prefixed error message.

### 6.4 Text Formatting (`formatText`)

Output is HTML-escaped first (`escapeHtml`), then a minimal Markdown-like transform is applied: `\n\n` → paragraph breaks, `\n` → `<br>`, `**bold**` → `<strong>`. This is a deliberately minimal renderer — no markdown library — sufficient for the LLM's typical answer formatting without introducing an XSS surface (escaping happens before any tag injection).

## 7. Debugging Utilities

### 7.1 `debug_pdf.py`

Standalone script (no FastAPI/Pinecone dependency) to verify PyMuPDF extraction in isolation, before touching the index. Scans `/tmp/rag_docs` (or a custom directory arg), loads each file, and prints character counts + text previews for the first 3 pages/chunks. Explicitly checks for the `%PDF` raw-bytes failure signature and prints a clear fix instruction if PyMuPDF isn't installed correctly. Useful as a first diagnostic step when a newly uploaded document produces no answers.

### 7.2 `diagnose.py`

More targeted: takes a single PDF path and runs it through the exact same guard sequence used in `pinecone_store._clean_doc_text` (marker stripping, UTF-8 cleaning, control-char replacement, non-Latin ratio check), printing per-page pass/fail results and which specific guard rejected each page. Also includes a Step 4 that bypasses LlamaIndex entirely and calls pymupdf directly, to isolate whether a failure is in the PyMuPDF layer or the LlamaIndex reader wrapper. This script is effectively the debugging tool that the non-Latin ratio guard logic itself was developed/tuned against.

## 8. Known Limitations & Edge Cases (as implemented)

- **Single-process, in-memory state:** `chat_history` and `indexed_files` do not survive a server restart (only the Pinecone vectors do); there is no multi-user session separation — all users of a running instance share one state dict.
- **`KNOWN_WORDS` in `_file_keywords`** is hardcoded to the test document set — cross-document detection quality will degrade on filenames using vocabulary outside that list unless it's extended.
- **OCR'd PDF page numbers can be approximate:** when Docling's per-item provenance isn't available, page-level citations fall back to sequential chunk numbers from naive 2000-char splitting, not true PDF page numbers.
- **`pinecone_index.delete(delete_all=True)`** is wrapped in try/except specifically because Pinecone throws on an empty namespace — this is a known client quirk, not a bug in this code.
- **CORS is fully open** (`allow_origins=["*"]`) — acceptable for local/internal demo use, should be restricted before any external deployment.
- **No formal accuracy evaluation is implemented** — `MIN_SCORE`, `TOP_K`, etc. were tuned empirically against real test documents, not validated against a golden Q&A dataset (RAGAS was identified as future work, not implemented).
- **`app.py` is dead code in production** — it duplicates most of `main.py`'s logic against a Streamlit UI instead of FastAPI/SSE, and should either be deleted or clearly marked archival to avoid confusion for a future maintainer about which file is authoritative.

---

Prepared by: Mannat Bansal
