"""
src/evaluation/evaluate_retrieval.py

Evaluates the Sentence-BERT Retrieval System on the ACORD test split.
Calculates Standard Information Retrieval (IR) Metrics:
- Recall@1
- Recall@5
- Recall@10
- Mean Reciprocal Rank (MRR)

Saves results to:
- outputs/reports/retrieval_metrics.json
- outputs/reports/retrieval_metrics.txt
"""

import os
import sys
import json
import torch
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import get_project_root, setup_logger, load_jsonl

logger = setup_logger("evaluate_retrieval")


def evaluate_acord_retrieval(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    root = get_project_root()
    processed_dir = root / "data" / "processed"
    models_dir = root / "outputs" / "models"
    reports_dir = root / "outputs" / "reports"
    os.makedirs(reports_dir, exist_ok=True)

    embeddings_path = models_dir / "acord_corpus_embeddings.pt"
    metadata_path = models_dir / "acord_corpus_metadata.json"
    queries_path = processed_dir / "acord_queries.jsonl"
    qrels_path = processed_dir / "acord_qrels.jsonl"

    if not embeddings_path.exists() or not metadata_path.exists():
        logger.info("Corpus embeddings not found. Building index first...")
        from src.training.train_retrieval_model import build_and_cache_acord_index
        build_and_cache_acord_index(model_name)

    # Load corpus metadata and embeddings
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    corpus_ids = metadata["clause_ids"]
    corpus_id_to_idx = {cid: idx for idx, cid in enumerate(corpus_ids)}
    corpus_embeddings = torch.load(embeddings_path, map_location="cpu")

    # Load queries
    all_queries = list(load_jsonl(queries_path))
    queries_dict = {q["query_id"]: q["query"] for q in all_queries}

    # Load test qrels (relevance judgments)
    # Relevance definition: rating >= 2 is positive
    all_qrels = list(load_jsonl(qrels_path))
    test_qrels = defaultdict(dict)
    for q in all_qrels:
        if q["split"] == "test":
            qid = q["query_id"]
            cid = q["clause_id"]
            rating = q["rating"]
            if rating >= 2:  # Only track positive target clauses
                test_qrels[qid][cid] = rating

    logger.info(f"Loaded {len(test_qrels)} test queries with positive target clauses.")

    model = SentenceTransformer(model_name)

    k_values = [1, 5, 10]
    recalls = {k: [] for k in k_values}
    reciprocal_ranks = []

    evaluated_queries = 0

    for qid, relevant_clauses in test_qrels.items():
        if not relevant_clauses:
            continue

        q_text = queries_dict.get(qid, qid)
        q_emb = model.encode(q_text, convert_to_tensor=True, normalize_embeddings=True).cpu()

        # Compute cosine similarity against all corpus clauses
        cos_scores = util.cos_sim(q_emb, corpus_embeddings)[0]
        top_k_indices = torch.topk(cos_scores, k=max(k_values)).indices.tolist()
        top_k_clause_ids = [corpus_ids[idx] for idx in top_k_indices]

        # Calculate Recall@K
        total_relevant = len(relevant_clauses)
        for k in k_values:
            retrieved_at_k = set(top_k_clause_ids[:k])
            hits = len(retrieved_at_k.intersection(relevant_clauses.keys()))
            recall_at_k = hits / total_relevant if total_relevant > 0 else 0.0
            recalls[k].append(recall_at_k)

        # Calculate MRR (First relevant clause rank)
        rr = 0.0
        for rank, cid in enumerate(top_k_clause_ids, 1):
            if cid in relevant_clauses:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        evaluated_queries += 1

    # Average metrics
    avg_recall_1 = float(np.mean(recalls[1])) if recalls[1] else 0.0
    avg_recall_5 = float(np.mean(recalls[5])) if recalls[5] else 0.0
    avg_recall_10 = float(np.mean(recalls[10])) if recalls[10] else 0.0
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    metrics = {
        "model_name": model_name,
        "evaluation_split": "test",
        "evaluated_test_queries": evaluated_queries,
        "total_corpus_clauses": len(corpus_ids),
        "recall_at_1": round(avg_recall_1, 4),
        "recall_at_5": round(avg_recall_5, 4),
        "recall_at_10": round(avg_recall_10, 4),
        "mrr": round(mrr, 4)
    }

    # Save metrics JSON
    json_path = reports_dir / "retrieval_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save metrics text report
    txt_path = reports_dir / "retrieval_metrics.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("ACORD SEMANTIC RETRIEVAL EVALUATION REPORT\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Split: Test ({evaluated_queries} queries evaluated against {len(corpus_ids)} clauses)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Recall@1:   {avg_recall_1:.4f} ({avg_recall_1*100:.2f}%)\n")
        f.write(f"Recall@5:   {avg_recall_5:.4f} ({avg_recall_5*100:.2f}%)\n")
        f.write(f"Recall@10:  {avg_recall_10:.4f} ({avg_recall_10*100:.2f}%)\n")
        f.write(f"MRR:        {mrr:.4f}\n")
        f.write("=" * 60 + "\n")

    logger.info(f"\nACORD Retrieval Metrics:\nRecall@1: {avg_recall_1:.4f} | Recall@5: {avg_recall_5:.4f} | Recall@10: {avg_recall_10:.4f} | MRR: {mrr:.4f}")
    logger.info(f"Reports saved to {txt_path} and {json_path}")
    return metrics


if __name__ == "__main__":
    evaluate_acord_retrieval()
