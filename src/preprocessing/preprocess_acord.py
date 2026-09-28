"""
src/preprocessing/preprocess_acord.py

Extracts, normalizes, and packages ACORD retrieval data.
- Loads queries.jsonl, corpus.jsonl, and relevance judgments (qrels/train.tsv, valid.tsv, test.tsv)
- Standardizes records into:
    {
      "query_id": "...",
      "query": "...",
      "clause_id": "...",
      "clause_text": "...",
      "relevance": 1/0,
      "rating": 0-4,
      "split": "train/valid/test",
      "source": "ACORD"
    }
- Saves:
    data/processed/acord_normalized.jsonl
    data/processed/acord_corpus.jsonl
    data/processed/acord_queries.jsonl
    data/processed/acord_qrels.jsonl
"""

import os
import sys
import json
from collections import Counter
from pathlib import Path
import pandas as pd

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import (
    get_acord_path,
    get_project_root,
    setup_logger,
    clean_legal_text,
    save_jsonl
)

logger = setup_logger("preprocess_acord")


def preprocess_acord(relevance_threshold: int = 2) -> dict:
    acord_dir = get_acord_path()
    project_root = get_project_root()
    processed_dir = project_root / "data" / "processed"
    os.makedirs(processed_dir, exist_ok=True)

    logger.info(f"Inspecting ACORD directory: {acord_dir}")

    # Locate queries, corpus, and qrels
    queries_file = None
    corpus_file = None
    qrels_files = {}

    for p in acord_dir.rglob("*.jsonl"):
        if "queries" in p.name:
            queries_file = p
        elif "corpus" in p.name:
            corpus_file = p

    for p in acord_dir.rglob("*.tsv"):
        if "train" in p.name:
            qrels_files["train"] = p
        elif "valid" in p.name:
            qrels_files["valid"] = p
        elif "test" in p.name:
            qrels_files["test"] = p

    if not queries_file or not corpus_file:
        raise FileNotFoundError(f"Could not locate queries.jsonl or corpus.jsonl in {acord_dir}")

    # 1. Load queries
    queries_dict = {}
    queries_records = []
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                qid = item.get("_id")
                qtext = clean_legal_text(item.get("text", ""))
                meta = item.get("metadata", {})
                queries_dict[qid] = {
                    "query_id": qid,
                    "query": qtext,
                    "category": meta.get("category", "unspecified"),
                    "split": meta.get("split", "unspecified")
                }
                queries_records.append(queries_dict[qid])

    logger.info(f"Loaded {len(queries_dict)} queries.")

    # 2. Load corpus clauses
    corpus_dict = {}
    corpus_records = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                cid = item.get("_id")
                ctext = clean_legal_text(item.get("text", ""))
                corpus_dict[cid] = ctext
                corpus_records.append({
                    "clause_id": cid,
                    "clause_text": ctext,
                    "source": "ACORD"
                })

    logger.info(f"Loaded {len(corpus_dict)} clauses.")

    # 3. Process qrels and create normalized records
    normalized_records = []
    qrels_records = []
    split_pair_counts = Counter()
    split_rel_counts = Counter()

    for split_name, qpath in qrels_files.items():
        logger.info(f"Reading qrels: {split_name} from {qpath.name}")
        df = pd.read_csv(qpath, sep="\t")
        for _, row in df.iterrows():
            qid = str(row["query-id"])
            cid = str(row["corpus-id"])
            rating = int(row["score"])
            # In IR, rating >= 2 represents relevant/usable clauses
            is_rel = 1 if rating >= relevance_threshold else 0

            split_pair_counts[split_name] += 1
            if is_rel:
                split_rel_counts[split_name] += 1

            q_info = queries_dict.get(qid, {"query": qid, "category": "unknown"})
            c_text = corpus_dict.get(cid, "")

            qrel_entry = {
                "query_id": qid,
                "clause_id": cid,
                "rating": rating,
                "relevance": is_rel,
                "split": split_name
            }
            qrels_records.append(qrel_entry)

            # For normalized pairs, focus on positive matches + sampled negatives
            # to keep the primary paired JSONL file efficient and ready for training
            normalized_records.append({
                "query_id": qid,
                "query": q_info["query"],
                "category": q_info.get("category", "unknown"),
                "clause_id": cid,
                "document_id": cid,  # Clause identifier treated as document/passage ID
                "clause_text": c_text,
                "rating": rating,
                "relevance": is_rel,
                "split": split_name,
                "source": "ACORD"
            })

    # Save output files
    out_norm = processed_dir / "acord_normalized.jsonl"
    out_queries = processed_dir / "acord_queries.jsonl"
    out_corpus = processed_dir / "acord_corpus.jsonl"
    out_qrels = processed_dir / "acord_qrels.jsonl"

    save_jsonl(normalized_records, out_norm)
    save_jsonl(queries_records, out_queries)
    save_jsonl(corpus_records, out_corpus)
    save_jsonl(qrels_records, out_qrels)

    summary = {
        "total_queries": len(queries_dict),
        "total_corpus_clauses": len(corpus_dict),
        "total_qrels_pairs": len(qrels_records),
        "split_pairs": dict(split_pair_counts),
        "split_relevant_pairs": dict(split_rel_counts),
        "relevance_threshold": relevance_threshold,
        "output_files": [str(out_norm), str(out_queries), str(out_corpus), str(out_qrels)]
    }

    logger.info(f"Preprocessed ACORD saved successfully.")
    return summary


if __name__ == "__main__":
    res = preprocess_acord()
    print("\nACORD Preprocessing Summary:")
    print(f"Total queries: {res['total_queries']}")
    print(f"Total clauses: {res['total_corpus_clauses']}")
    print(f"Total judgments: {res['total_qrels_pairs']}")
    print("Pairs by split:", res['split_pairs'])
    print("Relevant pairs (rating >= 2):", res['split_relevant_pairs'])
