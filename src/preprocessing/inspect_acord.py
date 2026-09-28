"""
src/preprocessing/inspect_acord.py

Deep inspection script for the ACORD (Atticus Clause Retrieval Dataset).
Inspects:
- File names, sizes, formats
- Query fields and count of unique queries
- Corpus fields and count of unique clauses
- Document/clause identifiers
- Relevance labels / rating fields across train/valid/test splits
- Distribution of relevance ratings
- Sample queries and clauses
"""

import os
import sys
import json
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import get_acord_path, setup_logger

logger = setup_logger("inspect_acord")


def inspect_acord(acord_dir: Path = None) -> dict:
    if acord_dir is None:
        acord_dir = get_acord_path()

    logger.info(f"Inspecting ACORD directory: {acord_dir}")

    # Discover files
    files_info = []
    for p in acord_dir.rglob("*"):
        if p.is_file():
            files_info.append({
                "name": p.name,
                "rel_path": str(p.relative_to(acord_dir)),
                "size_mb": round(p.stat().st_size / (1024 * 1024), 3),
                "extension": p.suffix
            })

    # Locate queries.jsonl, corpus.jsonl, and qrels
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

    logger.info(f"Queries file: {queries_file}")
    logger.info(f"Corpus file: {corpus_file}")

    # 1. Inspect queries
    queries = []
    query_fields = set()
    query_categories = Counter()
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                queries.append(item)
                query_fields.update(item.keys())
                meta = item.get("metadata", {})
                if isinstance(meta, dict) and "category" in meta:
                    query_categories[meta["category"]] += 1

    total_queries = len(queries)
    unique_query_ids = len(set(q.get("_id") for q in queries))
    query_sample = queries[:3] if queries else []

    # 2. Inspect corpus
    corpus = {}
    corpus_fields = set()
    clause_lengths_chars = []
    clause_lengths_words = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                cid = item.get("_id")
                txt = item.get("text", "")
                corpus[cid] = txt
                corpus_fields.update(item.keys())
                clause_lengths_chars.append(len(txt))
                clause_lengths_words.append(len(txt.split()))

    total_clauses = len(corpus)
    corpus_sample = list(corpus.items())[:3] if corpus else []

    # 3. Inspect qrels (train, valid, test)
    qrels_stats = {}
    total_pairs = 0
    all_scores = Counter()

    for split_name, qpath in qrels_files.items():
        df = pd.read_csv(qpath, sep="\t")
        total_split_pairs = len(df)
        total_pairs += total_split_pairs
        split_scores = Counter(df["score"].tolist())
        all_scores.update(split_scores)

        unique_q = df["query-id"].nunique()
        unique_c = df["corpus-id"].nunique()

        qrels_stats[split_name] = {
            "path": str(qpath.name),
            "columns": list(df.columns),
            "total_pairs": total_split_pairs,
            "unique_queries": unique_q,
            "unique_clauses": unique_c,
            "rating_distribution": dict(sorted(split_scores.items()))
        }

    results = {
        "dataset_name": "ACORD (Atticus Clause Retrieval Dataset)",
        "files": files_info,
        "query_fields": sorted(list(query_fields)),
        "total_unique_queries": total_queries,
        "query_categories": dict(query_categories.most_common(15)),
        "query_sample": query_sample,
        "corpus_fields": sorted(list(corpus_fields)),
        "total_corpus_clauses": total_clauses,
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
        },
        "qrels_stats": qrels_stats,
        "total_query_clause_pairs": total_pairs,
        "overall_rating_distribution": dict(sorted(all_scores.items()))
    }

    return results


def format_acord_report(results: dict) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("ACORD DATASET INSPECTION REPORT")
    lines.append("=" * 80)
    lines.append(f"Total Unique Queries: {results['total_unique_queries']}")
    lines.append(f"Query Fields: {results['query_fields']}")
    lines.append(f"Total Corpus Clauses: {results['total_corpus_clauses']}")
    lines.append(f"Corpus Fields: {results['corpus_fields']}")
    lines.append(f"Total Query-Clause Pairs Across Splits: {results['total_query_clause_pairs']}")
    lines.append("")

    lines.append("--- OVERALL RATING DISTRIBUTION ---")
    for score, cnt in results['overall_rating_distribution'].items():
        pct = (cnt / results['total_query_clause_pairs']) * 100 if results['total_query_clause_pairs'] else 0
        lines.append(f"  Rating {score}: {cnt:6d} pairs ({pct:5.2f}%)")
    lines.append("")

    lines.append("--- SPLIT DETAILS ---")
    for sname, sdata in results['qrels_stats'].items():
        lines.append(f"Split: {sname.upper()} ({sdata['path']})")
        lines.append(f"  Pairs: {sdata['total_pairs']}")
        lines.append(f"  Unique Queries: {sdata['unique_queries']}")
        lines.append(f"  Unique Clauses: {sdata['unique_clauses']}")
        lines.append(f"  Ratings: {sdata['rating_distribution']}")
    lines.append("")

    lines.append("--- CORPUS CLAUSE LENGTH STATISTICS ---")
    c_chars = results['clause_length_stats_chars']
    c_words = results['clause_length_stats_words']
    lines.append(f"Chars: Min = {c_chars['min']}, Mean = {c_chars['mean']}, Median = {c_chars['median']}, Max = {c_chars['max']}")
    lines.append(f"Words: Min = {c_words['min']}, Mean = {c_words['mean']}, Median = {c_words['median']}, Max = {c_words['max']}")
    lines.append("")

    lines.append("--- TOP QUERY CATEGORIES (Metadata) ---")
    for cat, cnt in list(results['query_categories'].items())[:10]:
        lines.append(f"  {cat:35} : {cnt} queries")
    lines.append("")

    lines.append("--- SAMPLE QUERIES ---")
    for q in results['query_sample']:
        lines.append(f"  ID: {q.get('_id')} | Text: \"{q.get('text')}\" | Category: {q.get('metadata', {}).get('category')}")
    lines.append("=" * 80)

    return "\n".join(lines)


if __name__ == "__main__":
    report = inspect_acord()
    formatted = format_acord_report(report)
    print(formatted)
