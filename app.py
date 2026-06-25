import os
import re
import json
import unicodedata
import datetime
import streamlit as st
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
ALLOWED_TYPES       = ["pdf", "txt", "md", "docx", "csv", "xlsx", "json", "pptx"]
MIN_SCORE           = 0.20

MAX_CHUNKS_PER_FILE = 5     # prevents any single large doc from dominating context
MAX_CHUNK_CHARS     = 6000  # total chars sent to LLM as context
MAX_HISTORY_CHARS   = 600   # chars of injected chat history
MAX_NON_LATIN       = 0.25  # corrupt-chunk guard
MAX_PROMPT_CHARS    = 11000 # hard cap before Groq API call

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 RAG Chatbot — Powered by Groq + LlamaIndex")

# ── LlamaIndex settings ─────────────────────────────────────────────────────────
def init_settings():
    Settings.llm = Groq(model=MODEL_NAME, api_key=GROQ_API_KEY)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    Settings.chunk_size    = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("retriever",     None),
    ("query_engine",  None),
    ("chat_history",  []),
    ("indexed_files", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Auto-load from ChromaDB on startup ─────────────────────────────────────────
if st.session_state.retriever is None:
    init_settings()
    index = load_index()
    if index:
        st.session_state.retriever    = index.as_retriever(similarity_top_k=TOP_K)
        st.session_state.query_engine = index.as_query_engine(
            similarity_top_k=TOP_K, response_mode="tree_summarize"
        )

# ── Text sanitiser ─────────────────────────────────────────────────────────────
def sanitise_text(text: str) -> str | None:
    """
    Layer 1: drop invalid UTF-8 bytes.
    Layer 2: replace Unicode control chars (except tab/newline) with spaces.
    Layer 3: non-Latin ratio guard — drops chunks from corrupt PDF font encoding
             (e.g. pages full of Kurdish/Arabic chars due to bad font maps).
    """
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

# ── Strip web-scrape annotation markers ───────────────────────────────────────
def strip_markers(text: str) -> str:
    """Remove |W| markers and [Source N — filename] lines from web-exported PDFs."""
    text = re.sub(r"\|W\|", " ", text)
    text = re.sub(r"\[Source\s*\d+\s*[—–-][^\]]*\]", "", text)
    return re.sub(r" {2,}", " ", text).strip()

# ── File keyword extractor (for cross-doc detection) ──────────────────────────
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

# ── Cross-document query detector ─────────────────────────────────────────────
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

# ── Multi-query retrieval for cross-document questions ─────────────────────────
def retrieve_cross_document(user_input: str) -> list:
    """
    Runs one focused sub-query per indexed file, then merges + deduplicates.
    Prevents a single embedding query from scoring low on cross-doc questions
    (where the query sits semantically between two document spaces).
    """
    indexed   = st.session_state.get("indexed_files", [])
    all_nodes = []
    seen_ids  = set()
    for fname in indexed:
        stem      = os.path.splitext(fname)[0].replace("_", " ")
        sub_query = f"{user_input} — focus on {stem}"
        try:
            nodes = st.session_state.retriever.retrieve(sub_query)
            for n in nodes:
                nid = id(n.node)
                if nid not in seen_ids:
                    seen_ids.add(nid)
                    all_nodes.append(n)
        except Exception:
            continue
    all_nodes.sort(key=lambda n: n.score or 0, reverse=True)
    return all_nodes

# ── Source citation block ──────────────────────────────────────────────────────
def format_sources(source_nodes) -> str:
    """
    Renders a clean Sources block. Only shows sources:
      - whose score is within 0.10 of the top score (relevance proximity filter)
      - whose filename is in indexed_files (ghost citation guard — no invented names)
    Shows at most 5 sources, sorted by relevance.
    """
    if not source_nodes:
        return ""

    indexed   = set(st.session_state.get("indexed_files", []))
    max_score = max((n.score for n in source_nodes if n.score), default=0)
    seen      = {}

    for node in source_nodes:
        meta  = node.node.metadata or {}
        fname = meta.get("file_name") or meta.get("filename") or ""
        if not fname or fname.strip() == "" or fname == "None":
            continue
        # Ghost citation guard
        if indexed:
            base = os.path.basename(fname)
            if fname not in indexed and base not in indexed:
                continue
        score = node.score or 0
        if max_score - score > 0.10:
            continue
        page    = meta.get("page_label") or meta.get("page_number") or meta.get("page") or None
        page = None if not page or str(page) == "None" else page
        score_r = round(score, 3)
        page = None if str(page) == "None" else page
        key  = (fname, str(page))
        if key not in seen or seen[key]["score"] < score_r:
            seen[key] = {"fname": fname, "page": page, "score": score_r}

    if not seen:
        return ""

    top_sources = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:5]
    parts = ["\n---\n📎 **Sources**"]
    for info in top_sources:
        page_str = f"p. {info['page']}" if info["page"] and str(info["page"]) != "None" else "location unknown"
        score_str = info["score"]
        parts.append(f"📄 **{info['fname']}** — {page_str}  *(relevance: {score_str})*")
    return "\n\n".join(parts)

# ── Query rewriter ─────────────────────────────────────────────────────────────
def build_standalone_query(user_input: str, history: list) -> str:
    """
    Rewrites ambiguous follow-up questions that use pronouns or deictic words.
    Only triggers when: prior history exists AND question is short AND contains a pronoun.
    Long self-contained questions always skip (no wasted API call).
    """
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
        (m["content"][:200] for m in reversed(history) if m["role"] == "assistant"),
        ""
    )
    rewrite_prompt = (
        f"The user is asking questions about documents.\n"
        f"Previous question: {last_q}\n"
        f"Previous answer (summary): {last_a}\n"
        f"Follow-up question: {user_input}\n\n"
        "The follow-up uses a pronoun or vague reference. "
        "Rewrite it as one specific, self-contained search query "
        "that clearly names the subject being referenced.\n"
        "Examples:\n"
        "  'Who made it?' after discussing Daily Sahayak "
        "-> 'Who is the founder or creator of Daily Sahayak?'\n"
        "  'What are its features?' after discussing Daily Sahayak "
        "-> 'What are the key features of Daily Sahayak?'\n"
        "  'What about JavaScript?' after discussing CSS requirements "
        "-> 'What are the JavaScript requirements for the capstone project?'\n"
        "Return ONLY the rewritten query — no quotes, no explanation."
    )
    try:
        rewritten = Settings.llm.complete(rewrite_prompt).text.strip()
        if (not rewritten
                or len(rewritten) > 300
                or "relationship between" in rewritten.lower()):
            return user_input
        return rewritten
    except Exception:
        return user_input

# ── Main streaming answer ──────────────────────────────────────────────────────
def stream_answer(user_input: str):
    """
    Pipeline:
    1. Rewrite ambiguous follow-up queries (pronoun guard).
    2. Detect cross-document questions → use per-file sub-queries + merge.
       Single-doc questions → standard retriever call.
    3. Filter by MIN_SCORE. Sanitise and strip artefact markers per chunk.
    4. Fallback: if nothing passes MIN_SCORE, retry at MIN_SCORE - 0.05.
    5. Build context with per-file chunk cap and total char cap.
    6. Stream LLM response with hard prompt size guard.
    7. Append source citations (ghost-citation-guarded).
    8. Store answer text in history (capped at 40 messages).
    """
    standalone    = build_standalone_query(user_input, st.session_state.chat_history)
    indexed_files = st.session_state.get("indexed_files", [])

    # Step 2: choose retrieval strategy
    if is_cross_document_query(user_input, indexed_files):
        all_nodes = retrieve_cross_document(standalone)
    else:
        all_nodes = st.session_state.retriever.retrieve(standalone)

    # Step 3: score filter + sanitise
    scored = [n for n in all_nodes if n.score is not None and n.score >= MIN_SCORE]
    source_nodes = []
    for n in scored:
        raw   = n.node.get_content()
        clean = sanitise_text(strip_markers(raw))
        if clean:
            n.node.set_content(clean)
            source_nodes.append(n)

    # Step 4: fallback — one step looser threshold
    if not source_nodes:
        FALLBACK_SCORE = max(0.10, MIN_SCORE - 0.05)
        scored_2 = [n for n in all_nodes if n.score is not None and n.score >= FALLBACK_SCORE]
        for n in scored_2:
            raw   = n.node.get_content()
            clean = sanitise_text(strip_markers(raw))
            if clean:
                n.node.set_content(clean)
                source_nodes.append(n)

    if not source_nodes:
        yield (
            "I couldn't find sufficiently relevant information in the uploaded documents "
            "to answer this confidently. Try rephrasing, or check that the relevant "
            "document has been indexed."
        )
        return

    # Step 5: build context with per-file cap + total char cap
    context_parts    = []
    total_chars      = 0
    file_chunk_count = {}

    for n in source_nodes:
        meta    = n.node.metadata or {}
        fname   = meta.get("file_name") or meta.get("filename") or "doc"
        page    = meta.get("page_label") or meta.get("page_number") or meta.get("page") or ""
        loc     = f"{fname}, p.{page}" if page else fname
        content = n.node.get_content()
        part    = f"[Source — {loc}]\n{content}"

        file_chunk_count[fname] = file_chunk_count.get(fname, 0)
        if file_chunk_count[fname] >= MAX_CHUNKS_PER_FILE:
            continue
        if total_chars + len(part) > MAX_CHUNK_CHARS:
            break

        context_parts.append(part)
        total_chars += len(part)
        file_chunk_count[fname] += 1

    if not context_parts:
        yield (
            "I found some relevant documents but couldn't fit any content within the "
            "context limit. Try asking a more specific question."
        )
        return

    context = "\n\n".join(context_parts)

    # History: last 2 user questions only, hard-capped at 600 chars
    history_text = ""
    recent_user  = [m for m in st.session_state.chat_history if m["role"] == "user"][-2:]
    if recent_user:
        raw_h        = "Recent questions:\n" + "\n".join(f"- {m['content']}" for m in recent_user) + "\n\n"
        history_text = raw_h[:MAX_HISTORY_CHARS]

    system_prompt = (
        "You are a document assistant. "
        "Answer ONLY using the text in the Sources provided below. "
        "Do NOT use general knowledge, training data, or outside information. "
        "Do NOT mention or invent any document name — only the exact filenames "
        "shown in the Sources. "
        "If the answer is not present in the Sources, say exactly: "
        "'This information is not found in the uploaded documents.' "
        "Never speculate. Never fill gaps with assumed knowledge."
    )

    full_prompt = (
        f"{system_prompt}\n\n"
        f"{history_text}"
        f"Sources:\n{context}\n\n"
        f"Question: {user_input}\n\nAnswer:"
    )

    # Step 6: hard prompt size cap
    if len(full_prompt) > MAX_PROMPT_CHARS:
        overflow    = len(full_prompt) - MAX_PROMPT_CHARS
        context     = context[: max(200, len(context) - overflow)]
        full_prompt = (
            f"{system_prompt}\n\n"
            f"{history_text}"
            f"Sources:\n{context}\n\n"
            f"Question: {user_input}\n\nAnswer:"
        )

    answer_text = ""
    try:
        for token in Settings.llm.stream_complete(full_prompt):
            if hasattr(token, "delta"):
                chunk = token.delta
            else:
                chunk = getattr(token, "text", None) or str(token)

            # Only yield non-empty strings — never None, never ""
            if not chunk or not isinstance(chunk, str):
                continue

            answer_text += chunk
            yield chunk
    except Exception as e:
        err = str(e)
        if "413" in err or "rate_limit" in err.lower() or "tokens" in err.lower():
            yield (
                "\n\n⚠️ The request was too large for the API limit. "
                "Try asking a more specific question, or clear and re-index fewer documents."
            )
        else:
            yield f"\n\n⚠️ Error: {err}"

    # Step 7: source citations (only when answer has real content)
    NOT_FOUND_PHRASE = "not found in the uploaded documents"
    if answer_text and NOT_FOUND_PHRASE not in answer_text:
        sources_block = format_sources(source_nodes)
        if sources_block:
            yield "\n\n" + sources_block

    # Step 8: store answer — cap history at 40 messages
    st.session_state.chat_history.append({"role": "assistant", "content": answer_text})
    if len(st.session_state.chat_history) > 40:
        st.session_state.chat_history = st.session_state.chat_history[-40:]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        f"Upload {', '.join(f'.{t}' for t in ALLOWED_TYPES)}",
        type=ALLOWED_TYPES,
        accept_multiple_files=True,
    )
    build_index_btn = st.button("🔨 Build Index", use_container_width=True)

    st.divider()
    count = st.session_state.get("chunk_count", get_chunk_count())
    if count > 0:
        st.success(f"💾 Pinecone: {count} chunks stored")
        if st.button("🗑️ Clear Database", use_container_width=True):
            clear_index()
            st.session_state.retriever     = None
            st.session_state.query_engine  = None
            st.session_state.chat_history  = []
            st.session_state.indexed_files = []
            st.rerun()
    else:
        st.info("💾 Pinecone: empty")

    if st.session_state.indexed_files:
        st.divider()
        st.subheader("📄 Indexed Files")
        for f in st.session_state.indexed_files:
            st.caption(f"• {f}")

    if st.session_state.chat_history:
        st.divider()
        export_data = json.dumps(st.session_state.chat_history, indent=2)
        st.download_button(
            "💬 Export Chat History",
            data=export_data,
            file_name=f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )

# ── Build index ────────────────────────────────────────────────────────────────
if build_index_btn:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one document.")
    else:
        oversized = [f.name for f in uploaded_files if f.size > MAX_FILE_SIZE]
        if oversized:
            st.sidebar.error(f"File(s) too large (max 20 MB): {', '.join(oversized)}")
        else:
            with st.spinner("Embedding documents and saving to ChromaDB..."):
                os.makedirs(TMP_DIR, exist_ok=True)
                saved_names = []
                for f in uploaded_files:
                    dest = os.path.join(TMP_DIR, f.name)
                    with open(dest, "wb") as out:
                        out.write(f.read())
                    saved_names.append(f.name)

                init_settings()
                try:
                    index = build_index()
                    st.session_state.retriever    = index.as_retriever(similarity_top_k=TOP_K)
                    st.session_state.query_engine = index.as_query_engine(
                        similarity_top_k=TOP_K, response_mode="tree_summarize"
                    )
                    st.session_state.chat_history  = []
                    st.session_state.indexed_files = saved_names
                    st.sidebar.success("✅ Index built and saved to ChromaDB!")
                    st.rerun()
                except ValueError as e:
                    st.sidebar.error(f"❌ Indexing failed:\n\n{e}")
st.session_state.chunk_count = get_chunk_count()
# ── Chat interface ─────────────────────────────────────────────────────────────
if st.session_state.retriever is None:
    st.info("👈 Upload documents and click **Build Index** to get started.")
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about your documents...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_text   = ""
            for chunk in stream_answer(user_input):
                if chunk and isinstance(chunk, str):
                    full_text += chunk
                    placeholder.markdown(full_text + "▌")
            placeholder.markdown(full_text)