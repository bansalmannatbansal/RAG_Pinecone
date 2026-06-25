"""
debug_pdf.py — Standalone script to verify PDF text extraction before indexing.

Run this BEFORE building the index to confirm that PyMuPDF is correctly
installed and that real document text (not raw PDF bytes) is being extracted.

Usage:
    python debug_pdf.py
    python debug_pdf.py /path/to/custom/dir   # optional custom directory

Expected output:
    FILE: Daily_Sahayak.pdf
    --- Page 1 ---
    Daily Sahayak
    Closing the Execution Gap
    ...

If you see '%PDF-1.7' or binary-looking output, PyMuPDF is not installed.
Fix: pip install pymupdf llama-index-readers-file
"""

import os
import sys
from pathlib import Path

# ── Directory to scan ──────────────────────────────────────────────────────────
TMP_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/rag_docs"

# ── Check PyMuPDF is installed ─────────────────────────────────────────────────
try:
    from llama_index.readers.file import PyMuPDFReader
    pdf_reader = PyMuPDFReader()
    print("✅ PyMuPDF is installed.\n")
except ImportError:
    print("❌ PyMuPDF is NOT installed.")
    print("   Fix: pip install pymupdf llama-index-readers-file")
    sys.exit(1)

from llama_index.core import SimpleDirectoryReader

# ── Scan directory ─────────────────────────────────────────────────────────────
if not os.path.isdir(TMP_DIR):
    print(f"❌ Directory not found: {TMP_DIR}")
    print("   Upload a document via the Streamlit UI first, then run this script.")
    sys.exit(1)

files = [f for f in os.listdir(TMP_DIR) if os.path.isfile(os.path.join(TMP_DIR, f))]
if not files:
    print(f"❌ No files found in {TMP_DIR}")
    print("   Upload a document via the Streamlit UI first, then run this script.")
    sys.exit(1)

print(f"Found {len(files)} file(s) in {TMP_DIR}\n")

for fname in files:
    fpath = os.path.join(TMP_DIR, fname)
    ext   = Path(fname).suffix.lower()

    print(f"{'=' * 60}")
    print(f"FILE: {fname}  ({ext})")
    print(f"{'=' * 60}")

    try:
        if ext == ".pdf":
            docs = pdf_reader.load_data(file=fpath)
        else:
            docs = SimpleDirectoryReader(input_files=[fpath]).load_data()

        if not docs:
            print("⚠️  No pages/chunks returned — file may be empty.\n")
            continue

        total_chars = sum(len(d.text) for d in docs)
        print(f"Pages/chunks extracted: {len(docs)}")
        print(f"Total characters: {total_chars}\n")

        for i, doc in enumerate(docs[:3], 1):
            preview = doc.text.strip()[:500]

            # Check for raw PDF bytes
            if preview.startswith("%PDF"):
                print(f"--- Page/chunk {i} ---")
                print("❌ RAW PDF BYTES DETECTED — extraction failed.")
                print("   This means PyMuPDF is not being used correctly.")
                print(f"   Raw preview: {preview[:100]!r}\n")
            else:
                print(f"--- Page/chunk {i} ---")
                print(preview)
                print()

    except Exception as e:
        print(f"❌ ERROR loading {fname}: {e}\n")

print("=" * 60)
print("Debug complete.")
print("If all previews show real text, your PDF extraction is working correctly.")
print("You can now rebuild the index via the Streamlit UI.")