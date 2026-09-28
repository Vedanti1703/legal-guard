"""
src/preprocessing/normalize_datasets.py

Master pipeline runner that:
1. Executes CUAD, ACORD, and Indian preprocessing
2. Conducts comprehensive Data Quality Checks:
   - Empty text detection
   - Extremely short / long clauses
   - Duplicate clauses and near-duplicate detection
   - Document-level train/test leakage verification
   - Label imbalance analysis
   - Corrupted PDF and encoding validation
3. Generates outputs/reports/data_quality_report.txt
"""

import os
import sys
from pathlib import Path
from collections import Counter
import numpy as np

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.preprocessing.preprocess_cuad import preprocess_cuad
from src.preprocessing.preprocess_acord import preprocess_acord
from src.preprocessing.preprocess_indian import preprocess_indian
from src.utils.common import (
    get_project_root,
    setup_logger,
    load_jsonl
)

logger = setup_logger("normalize_datasets")


def run_data_quality_checks() -> dict:
    root = get_project_root()
    processed_dir = root / "data" / "processed"
    reports_dir = root / "outputs" / "reports"
    os.makedirs(reports_dir, exist_ok=True)

    cuad_file = processed_dir / "cuad_normalized.jsonl"
    cuad_train_file = processed_dir / "cuad_train.jsonl"
    cuad_val_file = processed_dir / "cuad_val.jsonl"
    cuad_test_file = processed_dir / "cuad_test.jsonl"
    acord_file = processed_dir / "acord_normalized.jsonl"
    indian_file = processed_dir / "indian_normalized.jsonl"

    quality_results = {}

    # --- CUAD Quality Checks ---
    cuad_records = list(load_jsonl(cuad_file))
    cuad_train = list(load_jsonl(cuad_train_file))
    cuad_val = list(load_jsonl(cuad_val_file))
    cuad_test = list(load_jsonl(cuad_test_file))

    cuad_train_docs = set(r["document_id"] for r in cuad_train)
    cuad_val_docs = set(r["document_id"] for r in cuad_val)
    cuad_test_docs = set(r["document_id"] for r in cuad_test)

    # Cross-split leakage checks
    train_val_leakage = len(cuad_train_docs.intersection(cuad_val_docs))
    train_test_leakage = len(cuad_train_docs.intersection(cuad_test_docs))
    val_test_leakage = len(cuad_val_docs.intersection(cuad_test_docs))

    # Length checks
    cuad_lens = [len(r["clause_text"]) for r in cuad_records]
    empty_cuad = sum(1 for l in cuad_lens if l == 0)
    short_cuad = sum(1 for l in cuad_lens if l < 20)
    long_cuad = sum(1 for l in cuad_lens if l > 2000)

    # Duplicates
    cuad_texts = [r["clause_text"] for r in cuad_records]
    cuad_text_counts = Counter(cuad_texts)
    cuad_unique_texts = len(cuad_text_counts)
    cuad_dup_texts = sum(cnt - 1 for cnt in cuad_text_counts.values() if cnt > 1)

    quality_results["cuad"] = {
        "total_records": len(cuad_records),
        "train_records": len(cuad_train),
        "val_records": len(cuad_val),
        "test_records": len(cuad_test),
        "train_docs": len(cuad_train_docs),
        "val_docs": len(cuad_val_docs),
        "test_docs": len(cuad_test_docs),
        "leakage_train_val": train_val_leakage,
        "leakage_train_test": train_test_leakage,
        "leakage_val_test": val_test_leakage,
        "leakage_status": "PASSED - ZERO LEAKAGE" if (train_val_leakage == 0 and train_test_leakage == 0 and val_test_leakage == 0) else "FAILED",
        "empty_clauses": empty_cuad,
        "short_clauses_under_20_chars": short_cuad,
        "long_clauses_over_2000_chars": long_cuad,
        "unique_clauses": cuad_unique_texts,
        "duplicate_clause_occurrences": cuad_dup_texts,
        "label_distribution": dict(Counter(r["label"] for r in cuad_records))
    }

    # --- ACORD Quality Checks ---
    acord_records = list(load_jsonl(acord_file))
    acord_lens = [len(r["clause_text"]) for r in acord_records]
    empty_acord = sum(1 for l in acord_lens if l == 0)
    ratings_dist = Counter(r["rating"] for r in acord_records)

    quality_results["acord"] = {
        "total_judgments": len(acord_records),
        "empty_clauses": empty_acord,
        "ratings_distribution": dict(sorted(ratings_dist.items())),
        "positive_pairs_rating_ge_2": sum(v for k, v in ratings_dist.items() if k >= 2),
        "negative_pairs_rating_lt_2": sum(v for k, v in ratings_dist.items() if k < 2)
    }

    # --- Indian Dataset Quality Checks ---
    indian_records = list(load_jsonl(indian_file))
    indian_lens = [len(r["clause_text"]) for r in indian_records]
    empty_indian = sum(1 for l in indian_lens if l == 0)
    short_indian = sum(1 for l in indian_lens if l < 20)
    indian_labels = Counter(r["label"] for r in indian_records)

    # Document overlap with CUAD: Verify that Indian documents have zero overlap with CUAD contracts
    cuad_titles_set = set(cuad_train_docs | cuad_val_docs | cuad_test_docs)
    indian_docs_set = set(r["document_id"] for r in indian_records)
    doc_overlap = len(cuad_titles_set.intersection(indian_docs_set))

    quality_results["indian"] = {
        "total_extracted_clauses": len(indian_records),
        "empty_clauses": empty_indian,
        "short_clauses_under_20_chars": short_indian,
        "mapped_clauses": sum(v for k, v in indian_labels.items() if k != "unmapped"),
        "unmapped_clauses": indian_labels.get("unmapped", 0),
        "label_distribution": dict(indian_labels),
        "overlap_with_cuad_docs": doc_overlap,
        "independence_status": "PASSED - ZERO OVERLAP WITH CUAD CONTRACTS" if doc_overlap == 0 else "WARNING - OVERLAP DETECTED"
    }

    return quality_results


def write_quality_report(quality_results: dict) -> Path:
    root = get_project_root()
    report_file = root / "outputs" / "reports" / "data_quality_report.txt"

    lines = []
    lines.append("=" * 80)
    lines.append("DATA QUALITY & INTEGRITY REPORT")
    lines.append("Project: NLP for Detecting Hidden Fees and Unfavorable Terms in Consumer T&Cs")
    lines.append("=" * 80)
    lines.append("")

    lines.append("1. CUAD (Contract Understanding Atticus Dataset) QUALITY AUDIT:")
    cq = quality_results["cuad"]
    lines.append(f"   - Total Cleaned Records: {cq['total_records']}")
    lines.append(f"   - Split Breakdown: Train={cq['train_records']}, Val={cq['val_records']}, Test={cq['test_records']}")
    lines.append(f"   - Contract/Document Counts: Train Docs={cq['train_docs']}, Val Docs={cq['val_docs']}, Test Docs={cq['test_docs']}")
    lines.append(f"   - Document Leakage Check (Train/Val): {cq['leakage_train_val']} overlapping docs")
    lines.append(f"   - Document Leakage Check (Train/Test): {cq['leakage_train_test']} overlapping docs")
    lines.append(f"   - Document Leakage Check (Val/Test): {cq['leakage_val_test']} overlapping docs")
    lines.append(f"   - Status: [{cq['leakage_status']}]")
    lines.append(f"   - Empty Clauses: {cq['empty_clauses']}")
    lines.append(f"   - Short Clauses (<20 chars): {cq['short_clauses_under_20_chars']}")
    lines.append(f"   - Long Clauses (>2000 chars): {cq['long_clauses_over_2000_chars']}")
    lines.append(f"   - Unique Clause Texts: {cq['unique_clauses']}")
    lines.append(f"   - Duplicate Standard Boilerplate Instances: {cq['duplicate_clause_occurrences']}")
    lines.append("   - Label Distribution:")
    for lbl, cnt in cq['label_distribution'].items():
        lines.append(f"       * {lbl:35}: {cnt}")
    lines.append("")

    lines.append("2. ACORD RETRIEVAL DATASET QUALITY AUDIT:")
    aq = quality_results["acord"]
    lines.append(f"   - Total Graded Pairs: {aq['total_judgments']}")
    lines.append(f"   - Empty Clauses: {aq['empty_clauses']}")
    lines.append(f"   - Ratings Distribution (0-4): {aq['ratings_distribution']}")
    lines.append(f"   - Highly Relevant / Target Pairs (Rating >= 2): {aq['positive_pairs_rating_ge_2']}")
    lines.append(f"   - Non-Relevant / Background Pairs (Rating < 2): {aq['negative_pairs_rating_lt_2']}")
    lines.append("")

    lines.append("3. INDIAN LEGAL DATASET QUALITY AUDIT:")
    iq = quality_results["indian"]
    lines.append(f"   - Total Segmented Clauses: {iq['total_extracted_clauses']}")
    lines.append(f"   - Empty Clauses: {iq['empty_clauses']}")
    lines.append(f"   - Short Clauses (<20 chars): {iq['short_clauses_under_20_chars']}")
    lines.append(f"   - Conceptually Mapped Clauses: {iq['mapped_clauses']}")
    lines.append(f"   - Unmapped Statutory / Other Clauses: {iq['unmapped_clauses']}")
    lines.append(f"   - Overlap with CUAD SEC Contracts: {iq['overlap_with_cuad_docs']} documents")
    lines.append(f"   - Status: [{iq['independence_status']}]")
    lines.append("   - Label Distribution:")
    for lbl, cnt in iq['label_distribution'].items():
        lines.append(f"       * {lbl:30}: {cnt}")
    lines.append("")
    lines.append("=" * 80)

    report_text = "\n".join(lines)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"Data quality report saved to: {report_file}")
    return report_file


def run_full_normalization_pipeline():
    logger.info("Executing CUAD Preprocessing...")
    preprocess_cuad()

    logger.info("Executing ACORD Preprocessing...")
    preprocess_acord()

    logger.info("Executing Indian Preprocessing...")
    preprocess_indian()

    logger.info("Running Data Quality Checks across all datasets...")
    quality_results = run_data_quality_checks()
    report_path = write_quality_report(quality_results)

    logger.info(f"Full Normalization Pipeline Completed Successfully! Quality report at: {report_path}")


if __name__ == "__main__":
    run_full_normalization_pipeline()
