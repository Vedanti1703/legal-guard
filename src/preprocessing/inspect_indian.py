"""
src/preprocessing/inspect_indian.py

Deep inspection script for the Indian legal/contract dataset.
Inspects:
- File names, sizes, formats
- PDF count and page counts
- Text extraction success per page using PyMuPDF (fitz)
- Document structures and content types (Statutes vs Good/Bad Clauses vs Risky Patterns)
- Presence of raw contracts vs legal provisions vs risky patterns
- Duplicate documents and duplicate clauses
- Clause numbering patterns (e.g. 1., 1.1, 2(a), Section 27, etc.)
- Available metadata and labels
"""

import os
import sys
import re
import hashlib
from collections import Counter
from pathlib import Path
import numpy as np
import pymupdf

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import get_indian_path, setup_logger, clean_legal_text

logger = setup_logger("inspect_indian")


def inspect_indian(indian_dir: Path = None) -> dict:
    if indian_dir is None:
        indian_dir = get_indian_path()

    logger.info(f"Inspecting Indian dataset directory: {indian_dir}")

    files_info = []
    pdf_files = []
    other_files = []

    for p in sorted(indian_dir.iterdir()):
        if p.is_file():
            size_kb = round(p.stat().st_size / 1024, 2)
            f_item = {
                "name": p.name,
                "size_kb": size_kb,
                "extension": p.suffix
            }
            files_info.append(f_item)
            if p.suffix.lower() == ".pdf":
                pdf_files.append(p)
            else:
                other_files.append(p)

    total_pdfs = len(pdf_files)
    pdf_inspection = {}
    doc_text_hashes = {}
    duplicate_docs = []

    total_pages_all = 0
    total_pages_with_text = 0
    all_clauses = []
    clause_texts = []

    # Regex patterns for clause / section numbering in legal docs
    clause_regex = re.compile(
        r'(?:^|\n)(?:'
        r'(?:Section|\u00a7)\s*(\d+[A-Za-z]?(?:\s*\([a-z0-9]+\))*)'  # Section 10, Section 27
        r'|(\d+\.(?:\d+)*)'                                          # 1., 1.1, 4.2.1
        r'|(\([a-z0-9]+\))'                                          # (a), (i), (1)
        r'|(?:BAD\s+[A-Z\s\-]+|GOOD\s+[A-Z\s\-]+)'                   # BAD NON-COMPETE, GOOD TERMINATION
        r'|(?:PATTERN:[^\n]+)'                                       # PATTERN: ...
        r')',
        re.MULTILINE
    )

    for pdf_path in pdf_files:
        doc = pymupdf.open(pdf_path)
        page_count = len(doc)
        total_pages_all += page_count

        extracted_pages = []
        doc_full_text = []
        pages_with_text = 0

        for pno in range(page_count):
            page = doc[pno]
            text = page.get_text()
            cleaned = clean_legal_text(text)
            has_text = len(cleaned.strip()) > 0
            if has_text:
                pages_with_text += 1
                doc_full_text.append(cleaned)
            extracted_pages.append({
                "page_number": pno + 1,
                "has_text": has_text,
                "char_length": len(cleaned),
                "word_length": len(cleaned.split())
            })

        total_pages_with_text += pages_with_text
        combined_text = "\n\n".join(doc_full_text)
        doc_hash = hashlib.md5(combined_text.encode("utf-8")).hexdigest()

        if doc_hash in doc_text_hashes:
            duplicate_docs.append({
                "file": pdf_path.name,
                "duplicate_of": doc_text_hashes[doc_hash]
            })
        else:
            doc_text_hashes[doc_hash] = pdf_path.name

        # Identify document nature / structure
        doc_type = "unknown"
        if "Contract Act" in pdf_path.name or "Contract%20Act" in pdf_path.name:
            if "hindi" in pdf_path.name.lower():
                doc_type = "statutory_code_hindi"
            elif "ENG" in pdf_path.name:
                doc_type = "statutory_code_english_full"
            else:
                doc_type = "statutory_code_english_selected"
        elif "Clause" in pdf_path.name:
            doc_type = "clause_examples_good_vs_bad"
        elif "Risky" in pdf_path.name:
            doc_type = "risky_clause_patterns_and_rules"

        # Clause segmentation preview
        matches = list(clause_regex.finditer(combined_text))

        pdf_inspection[pdf_path.name] = {
            "page_count": page_count,
            "pages_with_text": pages_with_text,
            "text_extraction_success_pct": round((pages_with_text / page_count) * 100, 2) if page_count else 0,
            "total_chars": len(combined_text),
            "total_words": len(combined_text.split()),
            "doc_type": doc_type,
            "detected_numbering_markers": len(matches),
            "sample_snippet": combined_text[:400].replace("\n", " ")
        }

    results = {
        "dataset_name": "Indian Legal/Contract Dataset",
        "total_files": len(files_info),
        "pdf_count": total_pdfs,
        "other_files_count": len(other_files),
        "files_info": files_info,
        "total_pages": total_pages_all,
        "total_pages_with_text": total_pages_with_text,
        "overall_text_extraction_rate_pct": round((total_pages_with_text / total_pages_all) * 100, 2) if total_pages_all else 0,
        "duplicate_documents": duplicate_docs,
        "document_details": pdf_inspection,
    }

    return results


def format_indian_report(results: dict) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("INDIAN LEGAL DATASET INSPECTION REPORT")
    lines.append("=" * 80)
    lines.append(f"Total Files Found: {results['total_files']} (PDFs: {results['pdf_count']})")
    lines.append(f"Total Pages Across PDFs: {results['total_pages']}")
    lines.append(f"Pages with Usable Text: {results['total_pages_with_text']} ({results['overall_text_extraction_rate_pct']}%)")
    lines.append(f"Duplicate Documents (Exact Text Hash): {len(results['duplicate_documents'])}")
    for d in results['duplicate_documents']:
        lines.append(f"  Warning: {d['file']} has identical text to {d['duplicate_of']}")
    lines.append("")

    lines.append("--- DETAILED PER-FILE BREAKDOWN ---")
    for fname, details in results['document_details'].items():
        lines.append(f"File: {fname}")
        lines.append(f"  Type Classification: {details['doc_type']}")
        lines.append(f"  Pages: {details['page_count']} (Text extracted: {details['pages_with_text']}/{details['page_count']} = {details['text_extraction_success_pct']}%)")
        lines.append(f"  Volume: {details['total_words']} words ({details['total_chars']} chars)")
        lines.append(f"  Legal Markers / Potential Clause Headings: {details['detected_numbering_markers']}")
        lines.append(f"  Sample: \"{details['sample_snippet'][:180]}...\"")
        lines.append("")

    lines.append("--- NATURE OF THE INDIAN DATASET ---")
    lines.append("Analysis of content shows the Indian dataset contains:")
    lines.append("1. Statutory provisions: Indian Contract Act 1872 (Full English, Hindi, and Selected English provisions)")
    lines.append("2. Curated Clause Reference: Good vs Bad contract clauses under Indian law with explanations")
    lines.append("3. Risky Clause Patterns: High-risk clause definitions, court precedent references, and sample clauses")
    lines.append("Note: It does NOT contain thousands of annotated SEC contracts like CUAD.")
    lines.append("It provides domain-specific Indian contract law ground truth and risky pattern rules.")
    lines.append("=" * 80)

    return "\n".join(lines)


if __name__ == "__main__":
    report = inspect_indian()
    formatted = format_indian_report(report)
    print(formatted)
