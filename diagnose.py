"""
diagnose.py — Pinpoints exactly which guard is rejecting all pages.

Usage:
    python diagnose.py /path/to/your/file.pdf
"""

import sys
import re
import unicodedata
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python diagnose.py /path/to/file.pdf")
    sys.exit(1)

fpath = sys.argv[1]
MAX_NON_LATIN = 0.25

# ── Step 1: Try PyMuPDFReader ──────────────────────────────────────────────────
print("\n── Step 1: Load with PyMuPDFReader ──────────────────────────────────────")
try:
    from llama_index.readers.file import PyMuPDFReader
    reader = PyMuPDFReader()
    docs = reader.load_data(file=fpath)
    print(f"✅ Loaded {len(docs)} page(s)")
except Exception as e:
    print(f"❌ PyMuPDFReader failed: {e}")
    docs = []

# ── Step 2: Raw text check ─────────────────────────────────────────────────────
print("\n── Step 2: Raw text preview (first 3 pages) ─────────────────────────────")
for i, doc in enumerate(docs[:3]):
    text = doc.text.strip()
    print(f"\n  Page {i+1} ({len(text)} chars):")
    print(f"  {repr(text[:200])}")

# ── Step 3: Run each cleaner guard and report which one rejects pages ──────────
print("\n── Step 3: Cleaner guard analysis ───────────────────────────────────────")

passed = 0
rejected_empty  = 0
rejected_latin  = 0
rejected_pdf    = 0

for i, doc in enumerate(docs):
    text = doc.text

    # Guard 0: raw PDF bytes
    if text.strip().startswith("%PDF"):
        rejected_pdf += 1
        print(f"  Page {i+1}: ❌ RAW PDF BYTES")
        continue

    # Guards 1-2: marker stripping (non-destructive, just clean)
    text = re.sub(r"\|W\|", " ", text)
    text = re.sub(r"\[Source\s*\d+\s*[—–-][^\]]*\]", "", text)

    # Guard 3: UTF-8 cleaning
    text = text.encode("utf-8", errors="ignore").decode("utf-8")

    # Guard 4: control char replacement
    cleaned = []
    for ch in text:
        if unicodedata.category(ch) == "Cc" and ch not in ("\t", "\n"):
            cleaned.append(" ")
        else:
            cleaned.append(ch)
    text = re.sub(r" {2,}", " ", "".join(cleaned)).strip()

    if not text:
        rejected_empty += 1
        print(f"  Page {i+1}: ❌ EMPTY after cleaning")
        continue

    # Guard 5: non-Latin ratio
    non_latin = sum(1 for ch in text if ord(ch) > 0x024F and ch not in " \t\n\r")
    ratio = non_latin / len(text) if len(text) > 0 else 0

    if ratio > MAX_NON_LATIN:
        rejected_latin += 1
        # Show a sample of the non-Latin characters
        non_latin_chars = [ch for ch in text if ord(ch) > 0x024F and ch not in " \t\n\r"]
        print(f"  Page {i+1}: ❌ NON-LATIN RATIO {ratio:.2%}  "
              f"(sample: {non_latin_chars[:10]})")
        continue

    passed += 1
    if passed <= 2:
        print(f"  Page {i+1}: ✅ PASSED  ({len(text)} chars) — preview: {repr(text[:80])}")

print(f"\n── Summary ──────────────────────────────────────────────────────────────")
print(f"  Total pages  : {len(docs)}")
print(f"  ✅ Passed    : {passed}")
print(f"  ❌ Raw PDF   : {rejected_pdf}")
print(f"  ❌ Empty     : {rejected_empty}")
print(f"  ❌ Non-Latin : {rejected_latin}")

# ── Step 4: Try pymupdf directly (bypass LlamaIndex) ──────────────────────────
print("\n── Step 4: Direct pymupdf extraction (bypasses LlamaIndex) ─────────────")
try:
    import pymupdf  # fitz
    pdf = pymupdf.open(fpath)
    print(f"✅ pymupdf opened file — {pdf.page_count} page(s)")
    for i in range(min(3, pdf.page_count)):
        page = pdf[i]
        text = page.get_text().strip()
        print(f"\n  Page {i+1} ({len(text)} chars):")
        print(f"  {repr(text[:200])}")
    pdf.close()
except ImportError:
    print("❌ pymupdf not importable directly — try: pip install pymupdf")
except Exception as e:
    print(f"❌ pymupdf error: {e}")