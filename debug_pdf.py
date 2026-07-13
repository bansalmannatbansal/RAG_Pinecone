"""
check_page3.py — checks whether Data Heist.pdf page 3 was indexed,
and if so, what score it gets for a team-size query.
"""
import os
from dotenv import load_dotenv
load_dotenv()

from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pinecone_store import load_index

# Same settings init as main.py
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

index = load_index()
if not index:
    print("No index found — did you build it yet?")
    exit()

retriever = index.as_retriever(similarity_top_k=25)

print("\n--- Query: 'team size members' ---")
nodes = retriever.retrieve("team size members")
found_p3 = False
for n in nodes:
    meta = n.node.metadata or {}
    fname = meta.get("file_name") or meta.get("filename") or ""
    page  = meta.get("page_label") or meta.get("page_number") or "?"
    if "Data Heist" in fname:
        marker = " <-- PAGE 3" if str(page) == "3" else ""
        print(f"  page {page} | score {n.score:.4f} | {n.node.get_content()[:80]!r}{marker}")
        if str(page) == "3":
            found_p3 = True

print(f"\nPage 3 found in top-25 results: {found_p3}")

# Also directly check ALL chunks for Data Heist.pdf regardless of query,
# to see if page 3 was even indexed at all
print("\n--- All indexed Data Heist.pdf chunks (any page) ---")
all_nodes = retriever.retrieve("data heist rule book event")
pages_seen = set()
for n in all_nodes:
    meta = n.node.metadata or {}
    fname = meta.get("file_name") or meta.get("filename") or ""
    if "Data Heist" in fname:
        page = meta.get("page_label") or meta.get("page_number") or "?"
        pages_seen.add(str(page))
print("Pages seen across all Data Heist.pdf chunks in this retrieval:", sorted(pages_seen))