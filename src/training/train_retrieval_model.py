"""
src/training/train_retrieval_model.py

Builds and indexes the Semantic Dense Retrieval system for ACORD legal clauses.
- Uses Sentence-BERT (sentence-transformers/all-MiniLM-L6-v2) for legal clause retrieval
- Encodes all 3,931 ACORD corpus clauses into dense vector representations
- Saves corpus embeddings and ID index to disk to avoid repeated recomputation:
    outputs/models/acord_corpus_embeddings.pt
    outputs/models/acord_corpus_metadata.json
- Verifies embedding pipeline by testing consumer-risk search queries:
    "Find clauses related to automatic renewal."
    "Find clauses involving cancellation penalties."
    "Find clauses requiring advance notice before termination."
    "Find clauses involving additional payments."
    "Find clauses restricting refunds."
"""

import os
import sys
import json
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import get_project_root, setup_logger, load_jsonl

logger = setup_logger("train_retrieval_model")

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_and_cache_acord_index(model_name: str = DEFAULT_MODEL_NAME):
    root = get_project_root()
    corpus_file = root / "data" / "processed" / "acord_corpus.jsonl"
    models_dir = root / "outputs" / "models"
    os.makedirs(models_dir, exist_ok=True)

    embeddings_save_path = models_dir / "acord_corpus_embeddings.pt"
    metadata_save_path = models_dir / "acord_corpus_metadata.json"

    logger.info(f"Loading ACORD corpus from: {corpus_file}")
    corpus_records = list(load_jsonl(corpus_file))
    logger.info(f"Total corpus clauses to index: {len(corpus_records)}")

    clause_ids = [r["clause_id"] for r in corpus_records]
    clause_texts = [r["clause_text"] for r in corpus_records]

    logger.info(f"Loading Sentence-BERT model: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info(f"Encoding {len(clause_texts)} clauses into dense vectors (dim={model.get_sentence_embedding_dimension()})...")
    # Batch encode on CPU/GPU
    corpus_embeddings = model.encode(
        clause_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    logger.info(f"Embeddings shape: {corpus_embeddings.shape}")

    # Save embeddings tensor
    torch.save(corpus_embeddings.cpu(), embeddings_save_path)
    logger.info(f"Saved corpus embeddings to: {embeddings_save_path}")

    # Save metadata index
    metadata = {
        "model_name": model_name,
        "embedding_dim": model.get_sentence_embedding_dimension(),
        "total_clauses": len(clause_ids),
        "clause_ids": clause_ids,
        "clause_texts": clause_texts
    }
    with open(metadata_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    logger.info(f"Saved metadata index to: {metadata_save_path}")

    # Demo consumer query retrieval
    logger.info("\n--- TESTING RETRIEVAL ON CONSUMER CONTRACT QUERIES ---")
    test_queries = [
        "Find clauses related to automatic renewal.",
        "Find clauses involving cancellation penalties.",
        "Find clauses requiring advance notice before termination.",
        "Find clauses involving additional payments.",
        "Find clauses restricting refunds."
    ]

    for q in test_queries:
        q_emb = model.encode(q, convert_to_tensor=True, normalize_embeddings=True).cpu()
        cos_scores = util.cos_sim(q_emb, corpus_embeddings.cpu())[0]
        top_k = torch.topk(cos_scores, k=2)

        logger.info(f"\nQuery: \"{q}\"")
        for score, idx in zip(top_k.values, top_k.indices):
            idx = idx.item()
            logger.info(f"  Score: {score.item():.4f} | ID: {clause_ids[idx]}")
            logger.info(f"  Snippet: {clause_texts[idx][:180].replace(chr(10), ' ')}...")

    return {
        "status": "INDEX_BUILT_SUCCESSFULLY",
        "embeddings_path": str(embeddings_save_path),
        "metadata_path": str(metadata_save_path),
        "num_clauses": len(clause_ids)
    }


if __name__ == "__main__":
    build_and_cache_acord_index()
