"""
src/preprocessing/preprocess_cuad.py

Extracts, cleans, normalizes, and splits legal clauses from CUAD.
- Filters for the 6 core consumer-relevant categories (+ balanced negative background class 'other_clause')
- Performs strictly document-level train/validation/test splits (70/15/15, seed=42)
  to ensure zero data leakage between splits.
- Formats records into standard JSONL structure.
- Saves:
    data/processed/cuad_normalized.jsonl
    data/processed/cuad_train.jsonl
    data/processed/cuad_val.jsonl
    data/processed/cuad_test.jsonl
"""

import os
import sys
import json
import random
from collections import Counter
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import (
    get_cuad_path,
    get_project_root,
    setup_logger,
    clean_legal_text,
    save_jsonl
)

logger = setup_logger("preprocess_cuad")

TARGET_CATEGORIES = {
    "renewal term": "Renewal Term",
    "notice period to terminate renewal": "Notice Period To Terminate Renewal",
    "termination for convenience": "Termination For Convenience",
    "liquidated damages": "Liquidated Damages",
    "post-termination services": "Post-Termination Services",
    "revenue/profit sharing": "Revenue/Profit Sharing"
}


def preprocess_cuad(random_seed: int = 42, neg_ratio: float = 1.0) -> dict:
    random.seed(random_seed)
    cuad_dir = get_cuad_path()
    project_root = get_project_root()
    processed_dir = project_root / "data" / "processed"
    os.makedirs(processed_dir, exist_ok=True)

    cuad_json_candidates = list(cuad_dir.rglob("CUADv1.json"))
    if not cuad_json_candidates:
        raise FileNotFoundError(f"CUADv1.json not found in {cuad_dir}")
    cuad_file = cuad_json_candidates[0]

    logger.info(f"Loading raw CUAD data from: {cuad_file}")
    with open(cuad_file, "r", encoding="utf-8") as f:
        cuad_data = json.load(f)

    contracts = cuad_data.get("data", [])
    logger.info(f"Total contracts loaded: {len(contracts)}")

    # 1. Document-level split
    contract_titles = [c.get("title", f"doc_{i}") for i, c in enumerate(contracts)]
    # Deduplicate titles while maintaining deterministic order
    unique_titles = sorted(list(set(contract_titles)))
    random.shuffle(unique_titles)

    n_total = len(unique_titles)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    train_titles = set(unique_titles[:n_train])
    val_titles = set(unique_titles[n_train:n_train + n_val])
    test_titles = set(unique_titles[n_train + n_val:])

    logger.info(f"Document split (zero leakage): Train={len(train_titles)}, Val={len(val_titles)}, Test={len(test_titles)}")

    # Verify zero document overlap
    assert len(train_titles.intersection(val_titles)) == 0, "Leakage between train and val!"
    assert len(train_titles.intersection(test_titles)) == 0, "Leakage between train and test!"
    assert len(val_titles.intersection(test_titles)) == 0, "Leakage between val and test!"

    # 2. Extract clauses
    records = []
    other_candidates = []
    skipped_short = 0

    for contract in contracts:
        doc_id = contract.get("title", "unknown_doc")
        if doc_id in train_titles:
            split = "train"
        elif doc_id in val_titles:
            split = "val"
        else:
            split = "test"

        for para in contract.get("paragraphs", []):
            for qa in para.get("qas", []):
                q_text = qa.get("question", "")
                if 'related to "' in q_text:
                    cat_name = q_text.split('related to "')[1].split('"')[0].strip()
                else:
                    cat_name = q_text.strip()

                cat_key = cat_name.lower()
                is_target = cat_key in TARGET_CATEGORIES

                for ans in qa.get("answers", []):
                    ans_text = clean_legal_text(ans.get("text", ""))
                    # Filter out degenerate or empty extractions (< 15 chars)
                    if len(ans_text) < 15:
                        skipped_short += 1
                        continue

                    start_idx = ans.get("answer_start", -1)
                    end_idx = start_idx + len(ans_text) if start_idx >= 0 else -1

                    if is_target:
                        records.append({
                            "document_id": doc_id,
                            "clause_text": ans_text,
                            "label": TARGET_CATEGORIES[cat_key],
                            "start_offset": start_idx,
                            "end_offset": end_idx,
                            "split": split,
                            "source": "CUAD"
                        })
                    else:
                        other_candidates.append({
                            "document_id": doc_id,
                            "clause_text": ans_text,
                            "label": "other_clause",
                            "start_offset": start_idx,
                            "end_offset": end_idx,
                            "split": split,
                            "source": "CUAD"
                        })

    # Sample balanced background 'other_clause' per split
    target_count = len(records)
    target_by_split = Counter(r["split"] for r in records)
    logger.info(f"Target category positive annotations extracted: {target_count}")
    logger.info(f"Target count by split: {dict(target_by_split)}")

    for split in ["train", "val", "test"]:
        split_target_cnt = target_by_split[split]
        desired_neg = int(split_target_cnt * neg_ratio)
        split_candidates = [c for c in other_candidates if c["split"] == split]
        random.shuffle(split_candidates)
        chosen_neg = split_candidates[:desired_neg]
        records.extend(chosen_neg)

    random.shuffle(records)

    # 3. Save datasets
    train_records = [r for r in records if r["split"] == "train"]
    val_records = [r for r in records if r["split"] == "val"]
    test_records = [r for r in records if r["split"] == "test"]

    out_all = processed_dir / "cuad_normalized.jsonl"
    out_train = processed_dir / "cuad_train.jsonl"
    out_val = processed_dir / "cuad_val.jsonl"
    out_test = processed_dir / "cuad_test.jsonl"

    save_jsonl(records, out_all)
    save_jsonl(train_records, out_train)
    save_jsonl(val_records, out_val)
    save_jsonl(test_records, out_test)

    summary = {
        "total_records": len(records),
        "target_positive_records": target_count,
        "negative_background_records": len(records) - target_count,
        "train_count": len(train_records),
        "val_count": len(val_records),
        "test_count": len(test_records),
        "skipped_short_clauses": skipped_short,
        "label_distribution_all": dict(Counter(r["label"] for r in records)),
        "label_distribution_train": dict(Counter(r["label"] for r in train_records)),
        "label_distribution_test": dict(Counter(r["label"] for r in test_records)),
        "output_files": [str(out_all), str(out_train), str(out_val), str(out_test)]
    }

    logger.info(f"Preprocessed CUAD saved successfully. Total: {len(records)} records")
    return summary


if __name__ == "__main__":
    res = preprocess_cuad()
    print("\nCUAD Preprocessing Summary:")
    print(f"Total processed records: {res['total_records']}")
    print(f"Train / Val / Test: {res['train_count']} / {res['val_count']} / {res['test_count']}")
    print("Label distribution across all splits:")
    for lbl, count in res['label_distribution_all'].items():
        print(f"  {lbl:35}: {count}")
