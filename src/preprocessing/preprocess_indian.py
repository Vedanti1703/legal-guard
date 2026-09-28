"""
src/preprocessing/preprocess_indian.py

Extracts and segments legal clauses from the Indian legal dataset:
- Clause Type.pdf (Good vs. Bad clauses under Indian law)
- Risky Patterns clause.pdf (12 risk categories with examples and statutory rules)
- Indian Contract Act 1872 (Statutory provisions like Sec 10, 23, 27, 28, 73, 74)

Preserves:
- document_id
- source file
- page number
- clause_id (preserving legal hierarchy: e.g., Section 27, SEC_11_PAT_1, 1.1)
- clause_text
- category and risk metadata
- label mapped to unified schema (or "unmapped")
- source: "Indian"

Output:
data/processed/indian_normalized.jsonl
"""

import os
import sys
import re
from pathlib import Path
from collections import Counter
import pymupdf

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import (
    get_indian_path,
    get_project_root,
    setup_logger,
    clean_legal_text,
    load_json,
    save_jsonl
)

logger = setup_logger("preprocess_indian")


def parse_clause_type_pdf(pdf_path: Path, mapping: dict) -> list:
    """Parses Clause Type.pdf extracting Good vs Bad clause pairs across pages."""
    doc = pymupdf.open(pdf_path)
    records = []

    # Build full text while recording page offsets
    page_texts = []
    page_starts = []
    curr_len = 0
    for pno in range(len(doc)):
        t = doc[pno].get_text()
        page_texts.append(t)
        page_starts.append(curr_len)
        curr_len += len(t) + 1

    full_text = "\n".join(page_texts)

    def get_page_number(char_pos: int) -> int:
        for idx in range(len(page_starts) - 1, -1, -1):
            if char_pos >= page_starts[idx]:
                return idx + 1
        return 1

    # Split into sections by numeric headings: e.g. "1. NON-COMPETE CLAUSES"
    section_splits = list(re.finditer(r'(?:^|\n)\s*(\d+)\.\s+([A-Z\s\-]+CLAUSES?)\s*\n', full_text))

    for s_idx, s_match in enumerate(section_splits):
        sec_num = s_match.group(1).strip()
        sec_title = s_match.group(2).strip()
        s_start = s_match.end()
        s_end = section_splits[s_idx + 1].start() if s_idx + 1 < len(section_splits) else len(full_text)
        sec_chunk = full_text[s_start:s_end]

        # Extract Bad Clause
        bad_pattern = re.search(r'BAD\s+[A-Z\s\-]+:\s*\n+["\']?([^"\'\n]+(?:\n[^"\'\n]+)*?)["\']?\s*\n+WHY IT[\'’]S BAD:\s*\n+([\s\S]*?)(?=GOOD\s+[A-Z\s\-]+:|$)', sec_chunk)
        if bad_pattern:
            bad_clause = clean_legal_text(bad_pattern.group(1))
            why_bad = clean_legal_text(bad_pattern.group(2))
            pos = s_start + bad_pattern.start()
            page_no = get_page_number(pos)

            label = "unmapped"
            for k, v in mapping.get("indian_risk_to_unified", {}).items():
                if sec_title.lower() in k.lower():
                    label = v
                    break

            records.append({
                "document_id": "IND_CLAUSE_TYPE_GUIDE",
                "source_file": pdf_path.name,
                "page": page_no,
                "clause_id": f"IND_CT_SEC_{sec_num}_BAD",
                "clause_text": bad_clause,
                "category": sec_title,
                "clause_nature": "bad_clause",
                "risk_level": "HIGH",
                "explanation": why_bad[:300],
                "label": label,
                "source": "Indian"
            })

        # Extract Good Clause
        good_pattern = re.search(r'GOOD\s+[A-Z\s\-]+:\s*\n+["\']?([^"\'\n]+(?:\n[^"\'\n]+)*?)["\']?\s*\n+WHY IT[\'’]S GOOD:\s*\n+([\s\S]*?)(?=----------------|$)', sec_chunk)
        if good_pattern:
            good_clause = clean_legal_text(good_pattern.group(1))
            why_good = clean_legal_text(good_pattern.group(2))
            pos = s_start + good_pattern.start()
            page_no = get_page_number(pos)

            label = "unmapped"
            for k, v in mapping.get("indian_risk_to_unified", {}).items():
                if sec_title.lower() in k.lower():
                    label = v
                    break

            records.append({
                "document_id": "IND_CLAUSE_TYPE_GUIDE",
                "source_file": pdf_path.name,
                "page": page_no,
                "clause_id": f"IND_CT_SEC_{sec_num}_GOOD",
                "clause_text": good_clause,
                "category": sec_title,
                "clause_nature": "good_clause",
                "risk_level": "LOW",
                "explanation": why_good[:300],
                "label": label,
                "source": "Indian"
            })

    logger.info(f"Extracted {len(records)} clauses from {pdf_path.name}")
    return records


def parse_risky_patterns_pdf(pdf_path: Path, mapping: dict) -> list:
    """Parses Risky Patterns clause.pdf extracting patterns and clauses."""
    doc = pymupdf.open(pdf_path)
    records = []

    current_section = "General Risky Pattern"
    sec_num = "1"
    statutory_basis = "Indian Contract Act 1872"

    for pno in range(len(doc)):
        page = doc[pno]
        text = page.get_text()

        # Check section heading
        sec_match = re.search(r'SECTION\s+(\d+):\s*([A-Z\s\-]+CLAUSES?)', text)
        if sec_match:
            sec_num = sec_match.group(1).strip()
            current_section = f"SECTION {sec_num}: {sec_match.group(2).strip()}"

        # Check legal basis
        basis_match = re.search(r'LEGAL BASIS:\s*\n+([^\n]+(?:\n[^\n]+){1,3})', text)
        if basis_match:
            statutory_basis = clean_legal_text(basis_match.group(1))

        # Extract pattern blocks
        pattern_blocks = re.finditer(
            r'PATTERN:\s*([^\n]+)\s*\n+EXAMPLE:\s*["\']?([^"\']+)["\']?\s*\n+RISK:\s*([^\n]+(?:\n[^\n]+)?)\s*\n+RECOMMENDED:\s*([^\n]+)',
            text
        )

        for p_idx, pb in enumerate(pattern_blocks, 1):
            pat_desc = clean_legal_text(pb.group(1))
            clause_example = clean_legal_text(pb.group(2))
            risk_desc = clean_legal_text(pb.group(3))
            rec_desc = clean_legal_text(pb.group(4))

            cid = f"SEC_{sec_num}_PAT_{p_idx}"

            # Label mapping
            label = "unmapped"
            for k, v in mapping.get("indian_risk_to_unified", {}).items():
                if f"SECTION {sec_num}" in k:
                    label = v
                    break

            records.append({
                "document_id": "IND_RISKY_PATTERNS",
                "source_file": pdf_path.name,
                "page": pno + 1,
                "clause_id": cid,
                "clause_text": clause_example,
                "category": current_section,
                "risk_pattern": pat_desc,
                "risk_explanation": risk_desc,
                "statutory_basis": statutory_basis[:120],
                "recommended_clause": rec_desc,
                "risk_level": "HIGH",
                "label": label,
                "source": "Indian"
            })

    logger.info(f"Extracted {len(records)} risky pattern clauses from {pdf_path.name}")
    return records


def parse_statutory_act_pdf(pdf_path: Path) -> list:
    """Parses Indian Contract Act 1872 extracting sections as statutory benchmarks."""
    doc = pymupdf.open(pdf_path)
    records = []

    full_text_by_page = []
    for pno in range(len(doc)):
        full_text_by_page.append((pno + 1, doc[pno].get_text()))

    # Find section headings: e.g. "10. What agreements are contracts.—"
    # or "27. Agreement in restraint of trade, void.—"
    for page_num, ptext in full_text_by_page:
        # Regex matching numbered statutory sections
        matches = re.finditer(
            r'(?:^|\n)\s*(\d+[A-Za-z]?)\.\s+([^—\n\.-]+)[—\.-]\s*([^\n]+(?:\n[^\n]+){1,10})',
            ptext
        )
        for m in matches:
            sec_num = m.group(1).strip()
            sec_title = clean_legal_text(m.group(2))
            sec_body = clean_legal_text(m.group(3))
            combined_sec_text = f"Section {sec_num} ({sec_title}): {sec_body}"

            if len(sec_body) > 30:
                records.append({
                    "document_id": "IND_CONTRACT_ACT_1872",
                    "source_file": pdf_path.name,
                    "page": page_num,
                    "clause_id": f"Section_{sec_num}",
                    "clause_text": combined_sec_text,
                    "category": f"Statutory Provision: {sec_title}",
                    "risk_level": "STATUTE",
                    "label": "unmapped",
                    "source": "Indian"
                })

    logger.info(f"Extracted {len(records)} statutory sections from {pdf_path.name}")
    return records


def preprocess_indian() -> dict:
    indian_dir = get_indian_path()
    project_root = get_project_root()
    processed_dir = project_root / "data" / "processed"
    os.makedirs(processed_dir, exist_ok=True)

    mapping_file = processed_dir / "label_mapping.json"
    mapping = load_json(mapping_file) if mapping_file.exists() else {}

    all_indian_clauses = []

    # 1. Parse Clause Type.pdf
    ct_candidates = list(indian_dir.rglob("*Clause*Type*.pdf"))
    if ct_candidates:
        ct_clauses = parse_clause_type_pdf(ct_candidates[0], mapping)
        all_indian_clauses.extend(ct_clauses)

    # 2. Parse Risky Patterns clause.pdf
    rp_candidates = list(indian_dir.rglob("*Risky*Patterns*.pdf"))
    if rp_candidates:
        rp_clauses = parse_risky_patterns_pdf(rp_candidates[0], mapping)
        all_indian_clauses.extend(rp_clauses)

    # 3. Parse English Contract Act
    act_candidates = list(indian_dir.rglob("*Contract*Act*ENG*.pdf"))
    if not act_candidates:
        act_candidates = [p for p in indian_dir.rglob("*.pdf") if "hindi" not in p.name.lower() and "Contract" in p.name]

    if act_candidates:
        act_clauses = parse_statutory_act_pdf(act_candidates[0])
        # Deduplicate statutory sections if any
        seen_sec = set()
        for ac in act_clauses:
            if ac["clause_id"] not in seen_sec:
                seen_sec.add(ac["clause_id"])
                all_indian_clauses.append(ac)

    # Save to indian_normalized.jsonl
    out_file = processed_dir / "indian_normalized.jsonl"
    save_jsonl(all_indian_clauses, out_file)

    labels_dist = Counter(c["label"] for c in all_indian_clauses)
    mapped_count = sum(v for k, v in labels_dist.items() if k != "unmapped")
    unmapped_count = labels_dist.get("unmapped", 0)

    summary = {
        "total_indian_clauses": len(all_indian_clauses),
        "mapped_clauses": mapped_count,
        "unmapped_clauses": unmapped_count,
        "label_distribution": dict(labels_dist),
        "output_file": str(out_file)
    }

    logger.info(f"Indian preprocessing complete. Saved {len(all_indian_clauses)} records to {out_file}")
    return summary


if __name__ == "__main__":
    res = preprocess_indian()
    print("\nIndian Dataset Preprocessing Summary:")
    print(f"Total extracted clauses: {res['total_indian_clauses']}")
    print(f"Mapped clauses: {res['mapped_clauses']}")
    print(f"Unmapped clauses: {res['unmapped_clauses']}")
    print("Label distribution:")
    for lbl, count in res['label_distribution'].items():
        print(f"  {lbl:30}: {count}")
