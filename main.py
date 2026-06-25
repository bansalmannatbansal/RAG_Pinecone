"""
main.py — FastAPI backend for RAG Document Q&A Chatbot
Replaces Streamlit with a proper API + HTML/CSS/JS frontend.
"""

import os
import re
import json
import shutil
import unicodedata
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pinecone_store import build_index, load_index, get_chunk_count, clear_index, TMP_DIR

# ── Config ─────────────────────────────────────────────────────────────────────
GROQ_API_KEY        = os.getenv("GROQ_API_KEY")
MODEL_NAME          = "llama-3.3-70b-versatile"
TOP_K               = 25
CHUNK_SIZE          = 512
CHUNK_OVERLAP       = 50
MAX_FILE_SIZE       = 20 * 1024 * 1024
ALLOWED_TYPES       = {"pdf", "txt", "md", "docx", "csv", "xlsx", "json", "pptx"}
MIN_SCORE           = 0.20
MAX_CHUNKS_PER_FILE = 5
MAX_CHUNK_CHARS     = 6000
MAX_HISTORY_CHARS   = 600
MAX_NON_LATIN       = 0.25
MAX_PROMPT_CHARS    = 11000

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files (app.js, style.css) ──────────────────────────────────────────
# Place index.html in templates/, app.js and style.css in static/
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── In-memory state (replaces Streamlit session state) ─────────────────────────
state = {
    "retriever":     None,
    "indexed_files": [],
    "chat_history":  [],
    "chunk_count":   0,
}

# ── Init LlamaIndex settings (once) ───────────────────────────────────────────
def init_settings():
    # Access Settings._llm directly to avoid triggering LlamaIndex's
    # OpenAI default resolution before we've set our own LLM.
    if Settings._llm is None:
        Settings.llm = Groq(model=MODEL_NAME, api_key=GROQ_API_KEY)
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        Settings.chunk_size    = CHUNK_SIZE
        Settings.chunk_overlap = CHUNK_OVERLAP

# ── Load existing index on startup ─────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    init_settings()
    index = load_index()
    if index:
        state["retriever"]   = index.as_retriever(similarity_top_k=TOP_K)
        state["chunk_count"] = get_chunk_count()

# ── Text helpers ───────────────────────────────────────────────────────────────
def sanitise_text(text: str):
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    cleaned = []
    for ch in text:
        if unicodedata.category(ch) == "Cc" and ch not in ("\t", "\n"):
            cleaned.append(" ")
        else:
            cleaned.append(ch)
    text = re.sub(r" {2,}", " ", "".join(cleaned)).strip()
    if not text:
        return None
    non_latin = sum(1 for ch in text if ord(ch) > 0x024F and ch not in " \t\n\r")
    if len(text) > 0 and non_latin / len(text) > MAX_NON_LATIN:
        return None
    return text

def strip_markers(text: str) -> str:
    text = re.sub(r"\|W\|", " ", text)
    text = re.sub(r"\[Source\s*\d+\s*[—–-][^\]]*\]", "", text)
    return re.sub(r" {2,}", " ", text).strip()

def format_sources(source_nodes) -> list:
    if not source_nodes:
        return []
    indexed   = set(state["indexed_files"])
    max_score = max((n.score for n in source_nodes if n.score), default=0)
    seen      = {}
    for node in source_nodes:
        meta  = node.node.metadata or {}
        fname = meta.get("file_name") or meta.get("filename") or ""
        if not fname or fname.strip() == "" or fname == "None":
            continue
        if indexed:
            base = os.path.basename(fname)
            if fname not in indexed and base not in indexed:
                continue
        score = node.score or 0
        if max_score - score > 0.10:
            continue
        page  = meta.get("page_label") or meta.get("page_number") or meta.get("page") or None
        page  = None if not page or str(page) == "None" else page
        score_r = round(score, 3)
        key   = (fname, str(page))
        if key not in seen or seen[key]["score"] < score_r:
            seen[key] = {"fname": fname, "page": page, "score": score_r}
    if not seen:
        return []
    top = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:5]
    return [
        {
            "file": s["fname"],
            "page": str(s["page"]) if s["page"] else None,
            "score": s["score"],
        }
        for s in top
    ]

def _file_keywords(fname: str) -> list:
    """
    Extract short meaningful tokens from a filename for cross-doc query detection.
    Handles underscored names, camelCase, and concatenated words:
      'Daily_Sahayak_Closing_the_Execution_Gap.pdf' -> ['daily', 'sahayak', 'execution', 'gap']
      'thelinuxcommandline.pdf'                      -> ['linux', 'command', 'line']
    """
    STOPWORDS = {"the", "a", "an", "of", "in", "on", "at", "to", "for",
                 "and", "or", "is", "by", "be", "it", "as", "with", "from"}
    KNOWN_WORDS = ["linux", "command", "line", "daily", "sahayak", "data",
                   "heist", "web", "dev", "back", "link", "engine", "phase",
                   "capstone", "quiz", "major", "closing", "execution", "gap",
                   "project", "instructions", "backlink"]
    stem  = os.path.splitext(fname)[0]
    parts = re.split(r"[_\s]+", stem)
    raw_tokens = []
    for p in parts:
        p = re.sub(r"([a-z])([A-Z])", r"\1 \2", p).lower()
        raw_tokens.extend(p.split())
    # Split long concatenated tokens using known words as substrings
    expanded = []
    for tok in raw_tokens:
        if len(tok) > 8:
            found = [kw for kw in KNOWN_WORDS if kw in tok]
            expanded.extend(found) if found else expanded.append(tok)
        else:
            expanded.append(tok)
    return [t for t in expanded if len(t) > 2 and t not in STOPWORDS]

def is_cross_document_query(user_input: str, indexed_files: list) -> bool:
    """
    Returns True when the question spans multiple documents.
    Triggers on explicit cross-doc words (both/each/compare/vs) OR
    when keywords from 2+ different indexed files appear in the question.
    """
    if len(indexed_files) < 2:
        return False
    q_lower = user_input.lower()
    cross_words = {"both", "each", "across", "compare", "versus", "vs"}
    if cross_words & set(q_lower.split()):
        return True
    files_mentioned = sum(
        1 for f in indexed_files
        if any(kw in q_lower for kw in _file_keywords(f))
    )
    return files_mentioned >= 2

def retrieve_cross_document(user_input: str) -> list:
    """
    Runs one focused sub-query per indexed file, then merges + deduplicates.
    Prevents a single embedding query from scoring low on cross-doc questions
    (where the query sits semantically between two document spaces).
    """
    indexed   = state.get("indexed_files", [])
    all_nodes = []
    seen_ids  = set()
    for fname in indexed:
        stem      = os.path.splitext(fname)[0].replace("_", " ")
        sub_query = f"{user_input} — focus on {stem}"
        try:
            nodes = state["retriever"].retrieve(sub_query)
            for n in nodes:
                nid = id(n.node)
                if nid not in seen_ids:
                    seen_ids.add(nid)
                    all_nodes.append(n)
        except Exception:
            continue
    all_nodes.sort(key=lambda n: n.score or 0, reverse=True)
    return all_nodes

def build_standalone_query(user_input: str, history: list) -> str:
    user_turns = [m for m in history if m["role"] == "user"]
    if len(user_turns) < 1:
        return user_input
    short_pronouns = {"it", "that", "this", "they", "them", "he", "she",
                      "those", "these", "its", "their", "here", "made", "built"}
    words = set(user_input.lower().split())
    if not (words & short_pronouns):
        return user_input
    last_q = user_turns[-1]["content"]
    last_a = next(
        (m["content"][:200] for m in reversed(history) if m["role"] == "assistant"), ""
    )
    rewrite_prompt = (
        f"Previous question: {last_q}\n"
        f"Previous answer summary: {last_a}\n"
        f"Follow-up: {user_input}\n\n"
        "Rewrite the follow-up as a self-contained search query. "
        "Return ONLY the rewritten query."
    )
    try:
        rewritten = Settings.llm.complete(rewrite_prompt).text.strip()
        if not rewritten or len(rewritten) > 300:
            return user_input
        return rewritten
    except Exception:
        return user_input

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text())

@app.get("/status")
async def status():
    return {
        "indexed_files": state["indexed_files"],
        "chunk_count":   state["chunk_count"],
        "ready":         state["retriever"] is not None,
    }

@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    os.makedirs(TMP_DIR, exist_ok=True)
    saved = []
    errors = []
    for f in files:
        ext = Path(f.filename).suffix.lower().lstrip(".")
        if ext not in ALLOWED_TYPES:
            errors.append(f"{f.filename}: unsupported type")
            continue
        content = await f.read()
        if len(content) > MAX_FILE_SIZE:
            errors.append(f"{f.filename}: exceeds 20MB limit")
            continue
        dest = os.path.join(TMP_DIR, f.filename)
        with open(dest, "wb") as out:
            out.write(content)
        saved.append(f.filename)
    return {"saved": saved, "errors": errors}

@app.post("/build")
async def build():
    init_settings()
    try:
        index = build_index()
        state["retriever"]    = index.as_retriever(similarity_top_k=TOP_K)
        state["chunk_count"]  = get_chunk_count()
        # Track indexed files
        state["indexed_files"] = [
            f for f in os.listdir(TMP_DIR)
            if os.path.isfile(os.path.join(TMP_DIR, f))
        ]
        state["chat_history"] = []
        return {
            "success": True,
            "chunk_count": state["chunk_count"],
            "indexed_files": state["indexed_files"],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
async def clear():
    clear_index()
    state["retriever"]     = None
    state["indexed_files"] = []
    state["chat_history"]  = []
    state["chunk_count"]   = 0
    # Clear tmp dir
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    return {"success": True}

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    if state["retriever"] is None:
        raise HTTPException(status_code=400, detail="No index loaded. Upload and build first.")

    async def generate() -> AsyncGenerator[str, None]:
        user_input = req.message
        history    = state["chat_history"]

        # Add user message to history
        state["chat_history"].append({"role": "user", "content": user_input})

        # Query rewriting
        standalone = build_standalone_query(user_input, history)

        # Retrieval — use per-file sub-queries for cross-document questions
        try:
            if is_cross_document_query(standalone, state["indexed_files"]):
                all_nodes = retrieve_cross_document(standalone)
            else:
                all_nodes = state["retriever"].retrieve(standalone)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            return

        # Score filter
        scored = [n for n in all_nodes if n.score is not None and n.score >= MIN_SCORE]
        source_nodes = []
        for n in scored:
            raw   = n.node.get_content()
            clean = sanitise_text(strip_markers(raw))
            if clean:
                n.node.set_content(clean)
                source_nodes.append(n)

        # Fallback
        if not source_nodes:
            fallback = max(0.10, MIN_SCORE - 0.05)
            for n in all_nodes:
                if n.score is not None and n.score >= fallback:
                    raw   = n.node.get_content()
                    clean = sanitise_text(strip_markers(raw))
                    if clean:
                        n.node.set_content(clean)
                        source_nodes.append(n)

        if not source_nodes:
            msg = "I couldn't find sufficiently relevant information in the uploaded documents."
            state["chat_history"].append({"role": "assistant", "content": msg})
            yield f"data: {json.dumps({'type': 'text', 'content': msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'sources': []})}\n\n"
            return

        # Build context — deduplicate near-identical chunks
        context_parts     = []
        total_chars       = 0
        file_chunk_count  = {}
        seen_fingerprints = set()

        for n in source_nodes:
            meta    = n.node.metadata or {}
            fname   = meta.get("file_name") or meta.get("filename") or "doc"
            page    = meta.get("page_label") or meta.get("page_number") or ""
            loc     = f"{fname}, p.{page}" if page else fname
            content = n.node.get_content()
            part    = f"[Source — {loc}]\n{content}"

            # Skip chunks whose first 120 chars match a chunk already included
            fp = content.strip()[:60]
            if fp in seen_fingerprints:
                continue
            seen_fingerprints.add(fp)

            if fname not in file_chunk_count:
                file_chunk_count[fname] = 0
            if file_chunk_count[fname] >= MAX_CHUNKS_PER_FILE:
                continue
            if total_chars + len(part) > MAX_CHUNK_CHARS:
                break
            context_parts.append(part)
            total_chars += len(part)
            file_chunk_count[fname] += 1

        if not context_parts:
            msg = "Found documents but couldn't fit content in context. Ask a more specific question."
            state["chat_history"].append({"role": "assistant", "content": msg})
            yield f"data: {json.dumps({'type': 'text', 'content': msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'sources': []})}\n\n"
            return

        # Deduplicate sentences across chunks before sending to LLM
        seen_sentences = set()
        deduped_parts = []
        for part in context_parts:
            lines = part.split('\n')
            header = lines[0]  # Keep the [Source — ...] header
            deduped_lines = [header]
            for line in lines[1:]:
                stripped = line.strip()
                if len(stripped) < 20 or stripped not in seen_sentences:
                    deduped_lines.append(line)
                    if len(stripped) >= 20:
                        seen_sentences.add(stripped)
            deduped_parts.append('\n'.join(deduped_lines))
        context = "\n\n".join(deduped_parts)
        print("=== CONTEXT SENT TO LLM ===")
        print(context[:2000])
        print("=== END CONTEXT ===")

        # History
        recent_user  = [m for m in history if m["role"] == "user"][-2:]
        history_text = ""
        if recent_user:
            raw_h        = "Recent questions:\n" + "\n".join(f"- {m['content']}" for m in recent_user) + "\n\n"
            history_text = raw_h[:MAX_HISTORY_CHARS]

        system_prompt = (
            "You are a document assistant. "
            "Answer ONLY using the text in the Sources provided below. "
            "Do NOT use general knowledge or outside information. "
            "Do NOT repeat yourself — state each point ONCE only. "
            "Write a single unified answer even if the same information appears in multiple sources. "
            "If the answer is not present, say exactly: "
            "'This information is not found in the uploaded documents.' "
            "Never speculate."
        )

        full_prompt = (
            f"{system_prompt}\n\n"
            f"{history_text}"
            f"Sources:\n{context}\n\n"
            f"Question: {user_input}\n\nAnswer:"
        )

        if len(full_prompt) > MAX_PROMPT_CHARS:
            overflow    = len(full_prompt) - MAX_PROMPT_CHARS
            context     = context[: max(200, len(context) - overflow)]
            full_prompt = (
                f"{system_prompt}\n\n"
                f"{history_text}"
                f"Sources:\n{context}\n\n"
                f"Question: {user_input}\n\nAnswer:"
            )

        # Stream response
        answer_text = ""
        try:
            for token in Settings.llm.stream_complete(full_prompt):
                chunk = getattr(token, "delta", None)
                if not chunk:
                    continue
                if not chunk or not isinstance(chunk, str):
                    continue
                answer_text += chunk
                yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"
        except Exception as e:
            err = str(e)
            yield f"data: {json.dumps({'type': 'error', 'content': err})}\n\n"

        # Sources
        NOT_FOUND = "not found in the uploaded documents"
        sources   = []
        if answer_text and NOT_FOUND not in answer_text:
            sources = format_sources(source_nodes)

        state["chat_history"].append({"role": "assistant", "content": answer_text})
        if len(state["chat_history"]) > 40:
            state["chat_history"] = state["chat_history"][-40:]

        yield f"data: {json.dumps({'type': 'done', 'sources': sources})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/history")
async def history():
    return {"history": state["chat_history"]}

@app.post("/clear-history")
async def clear_history():
    state["chat_history"] = []
    return {"success": True}