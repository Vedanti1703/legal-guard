"""
src/preprocessing/inspect_cuad.py

Deep inspection script for the CUAD (Contract Understanding Atticus Dataset).
Inspects:
- File names and formats
- Number of contracts
- Number of annotations
- All 41 available legal clause categories
- Number of examples per category
- Text length statistics (contracts & annotated clauses)
- Missing values / unanswerable questions
- Duplicate contracts and duplicate clauses
- Specific verification of target categories for consumer risk
"""

import os
import sys
import json
import hashlib
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import get_cuad_path, setup_logger

logger = setup_logger("inspect_cuad")


def inspect_cuad(cuad_dir: Path = None) -> dict:
    if cuad_dir is None:
        cuad_dir = get_cuad_path()

    logger.info(f"Inspecting CUAD directory: {cuad_dir}")

    # 1. Discover files
    files_info = []
    data_dir = cuad_dir / "data" if (cuad_dir / "data").exists() else cuad_dir
    for p in cuad_dir.rglob("*"):
        if p.is_file():
            files_info.append({
                "name": p.name,
                "rel_path": str(p.relative_to(cuad_dir)),
                "size_mb": round(p.stat().st_size / (1024 * 1024), 3),
                "extension": p.suffix
            })

    # Look for CUAD JSON file
    cuad_json_candidates = list(cuad_dir.rglob("CUADv1.json"))
    if not cuad_json_candidates:
        raise FileNotFoundError(f"CUADv1.json not found in {cuad_dir}")

    cuad_json_path = cuad_json_candidates[0]
    logger.info(f"Loading CUAD primary file: {cuad_json_path} (size: {cuad_json_path.stat().st_size / (1024*1024):.2f} MB)")

    with open(cuad_json_path, "r", encoding="utf-8") as f:
        cuad_data = json.load(f)

    contracts = cuad_data.get("data", [])
    num_contracts = len(contracts)

    # 2. Check category descriptions CSV if available
    category_desc_candidates = list(cuad_dir.rglob("category_descriptions.csv"))
    official_categories = []
    if category_desc_candidates:
        cat_df = pd.read_csv(category_desc_candidates[0])
        # Find column that holds category name
        col = [c for c in cat_df.columns if "Category" in c][0]
        official_categories = [c.replace("Category: ", "").strip() for c in cat_df[col].dropna().tolist()]

    # 3. Analyze contracts and annotations
    contract_titles = []
    contract_hashes = set()
    duplicate_contract_titles = []
    duplicate_contract_texts = 0
    contract_lengths_chars = []
    contract_lengths_words = []

    label_counts = Counter()
    unanswerable_per_label = Counter()
    clause_lengths_chars = []
    clause_lengths_words = []
    clause_texts = []
    annotations_list = []

    for contract_idx, contract in enumerate(contracts):
        title = contract.get("title", f"doc_{contract_idx}")
        if title in contract_titles:
            duplicate_contract_titles.append(title)
        contract_titles.append(title)

        paragraphs = contract.get("paragraphs", [])
        if not paragraphs:
            continue

        for p_idx, para in enumerate(paragraphs):
            context = para.get("context", "")
            chash = hashlib.md5(context.encode("utf-8")).hexdigest()
            if chash in contract_hashes:
                duplicate_contract_texts += 1
            else:
                contract_hashes.add(chash)

            char_len = len(context)
            word_len = len(context.split())
            contract_lengths_chars.append(char_len)
            contract_lengths_words.append(word_len)

            qas = para.get("qas", [])
            for qa in qas:
                q_text = qa.get("question", "")
                qa_id = qa.get("id", "")
                is_impossible = qa.get("is_impossible", False)
                answers = qa.get("answers", [])

                # Extract standard category name from question or id
                # Questions typically formatted as:
                # 'Highlight the parts (if any) of this contract related to "Category Name" that should be reviewed by a lawyer. Details: ...'
                if 'related to "' in q_text:
                    label = q_text.split('related to "')[1].split('"')[0].strip()
                else:
                    label = q_text

                if not answers or is_impossible:
                    unanswerable_per_label[label] += 1
                else:
                    label_counts[label] += len(answers)
                    for ans in answers:
                        ans_text = ans.get("text", "")
                        ans_start = ans.get("answer_start", -1)
                        clause_texts.append(ans_text)
                        clause_lengths_chars.append(len(ans_text))
                        clause_lengths_words.append(len(ans_text.split()))
                        annotations_list.append({
                            "document_id": title,
                            "label": label,
                            "text": ans_text,
                            "start": ans_start,
                            "end": ans_start + len(ans_text)
                        })

    # Clause duplicate analysis
    clause_counter = Counter(clause_texts)
    unique_clauses = len(clause_counter)
    exact_duplicate_clauses = sum(count - 1 for count in clause_counter.values() if count > 1)

    # Focus categories specified by project (support flexible capitalization)
    target_categories = [
        "Renewal Term",
        "Notice Period to Terminate Renewal",
        "Termination for Convenience",
        "Liquidated Damages",
        "Post-Termination Services",
        "Revenue/Profit Sharing"
    ]

    # Map normalized lower-case names to actual keys
    label_counts_lower = {k.lower(): (k, v) for k, v in label_counts.items()}
    unanswerable_lower = {k.lower(): (k, v) for k, v in unanswerable_per_label.items()}

    target_presence = {}
    for cat in target_categories:
        exact_match = None
        count = 0
        unans = 0
        if cat.lower() in label_counts_lower:
            exact_match, count = label_counts_lower[cat.lower()]
        if cat.lower() in unanswerable_lower:
            _, unans = unanswerable_lower[cat.lower()]

        target_presence[cat] = {
            "found_in_labels": exact_match is not None,
            "actual_label_name": exact_match if exact_match else "None",
            "annotation_count": count,
            "unanswerable_questions": unans
        }

    results = {
        "dataset_name": "CUAD (Contract Understanding Atticus Dataset)",
        "files": files_info,
        "primary_file": str(cuad_json_path),
        "total_contracts": num_contracts,
        "unique_contract_titles": len(set(contract_titles)),
        "duplicate_contract_titles": duplicate_contract_titles,
        "duplicate_contract_texts": duplicate_contract_texts,
        "total_annotations": len(clause_texts),
        "unique_clause_texts": unique_clauses,
        "duplicate_clause_occurrences": exact_duplicate_clauses,
        "total_categories": len(set(list(label_counts.keys()) + list(unanswerable_per_label.keys()))),
        "all_labels": sorted(list(set(list(label_counts.keys()) + list(unanswerable_per_label.keys())))),
        "label_counts": dict(label_counts.most_common()),
        "target_categories_verification": target_presence,
        "contract_length_stats_chars": {
            "min": int(np.min(contract_lengths_chars)) if contract_lengths_chars else 0,
            "mean": round(float(np.mean(contract_lengths_chars)), 2) if contract_lengths_chars else 0,
            "median": round(float(np.median(contract_lengths_chars)), 2) if contract_lengths_chars else 0,
            "max": int(np.max(contract_lengths_chars)) if contract_lengths_chars else 0
        },
        "contract_length_stats_words": {
            "min": int(np.min(contract_lengths_words)) if contract_lengths_words else 0,
            "mean": round(float(np.mean(contract_lengths_words)), 2) if contract_lengths_words else 0,
            "median": round(float(np.median(contract_lengths_words)), 2) if contract_lengths_words else 0,
            "max": int(np.max(contract_lengths_words)) if contract_lengths_words else 0
        },
        "clause_length_stats_chars": {
            "min": int(np.min(clause_lengths_chars)) if clause_lengths_chars else 0,
            "mean": round(float(np.mean(clause_lengths_chars)), 2) if clause_lengths_chars else 0,
            "median": round(float(np.median(clause_lengths_chars)), 2) if clause_lengths_chars else 0,
            "max": int(np.max(clause_lengths_chars)) if clause_lengths_chars else 0
        },
        "clause_length_stats_words": {
            "min": int(np.min(clause_lengths_words)) if clause_lengths_words else 0,
            "mean": round(float(np.mean(clause_lengths_words)), 2) if clause_lengths_words else 0,
            "median": round(float(np.median(clause_lengths_words)), 2) if clause_lengths_words else 0,
            "max": int(np.max(clause_lengths_words)) if clause_lengths_words else 0
        }
    }

    return results


def format_cuad_report(results: dict) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("CUAD DATASET INSPECTION REPORT")
    lines.append("=" * 80)
    lines.append(f"Total Contracts: {results['total_contracts']}")
    lines.append(f"Unique Titles: {results['unique_contract_titles']}")
    lines.append(f"Duplicate Contract Titles: {len(results['duplicate_contract_titles'])}")
    lines.append(f"Duplicate Contract Texts (exact MD5 hash match): {results['duplicate_contract_texts']}")
    lines.append(f"Total Annotations (Positive Clause Answers): {results['total_annotations']}")
    lines.append(f"Unique Annotated Clauses: {results['unique_clause_texts']}")
    lines.append(f"Duplicate Clause Instances: {results['duplicate_clause_occurrences']}")
    lines.append(f"Total Categories Identified: {results['total_categories']}")
    lines.append("")

    lines.append("--- CONTRACT LENGTH STATISTICS ---")
    c_chars = results['contract_length_stats_chars']
    c_words = results['contract_length_stats_words']
    lines.append(f"Chars: Min = {c_chars['min']}, Mean = {c_chars['mean']}, Median = {c_chars['median']}, Max = {c_chars['max']}")
    lines.append(f"Words: Min = {c_words['min']}, Mean = {c_words['mean']}, Median = {c_words['median']}, Max = {c_words['max']}")
    lines.append("")

    lines.append("--- CLAUSE LENGTH STATISTICS ---")
    cl_chars = results['clause_length_stats_chars']
    cl_words = results['clause_length_stats_words']
    lines.append(f"Chars: Min = {cl_chars['min']}, Mean = {cl_chars['mean']}, Median = {cl_chars['median']}, Max = {cl_chars['max']}")
    lines.append(f"Words: Min = {cl_words['min']}, Mean = {cl_words['mean']}, Median = {cl_words['median']}, Max = {cl_words['max']}")
    lines.append("")

    lines.append("--- TARGET CONSUMER-RISK CATEGORIES VERIFICATION ---")
    for cat, info in results['target_categories_verification'].items():
        status = "FOUND" if info['found_in_labels'] else "NOT FOUND"
        actual = f" (in CUAD as '{info['actual_label_name']}')" if info['found_in_labels'] else ""
        lines.append(f"[{status}] {cat:38}: {info['annotation_count']:5} annotations{actual}")
    lines.append("")

    lines.append("--- ALL 41 CUAD CATEGORIES (Ranked by Annotation Count) ---")
    for rank, (label, count) in enumerate(results['label_counts'].items(), 1):
        lines.append(f"{rank:2d}. {label:38} : {count:5} examples")
    lines.append("=" * 80)

    return "\n".join(lines)


if __name__ == "__main__":
    report = inspect_cuad()
    formatted = format_cuad_report(report)
    print(formatted)
