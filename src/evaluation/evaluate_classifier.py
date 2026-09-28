"""
src/evaluation/evaluate_classifier.py

Unified evaluation harness for Legal Clause Classifiers on held-out test splits.
- Evaluates saved models:
    outputs/models/tfidf_baseline.joblib
    outputs/models/legal_bert_clause_classifier/
- Generates:
    outputs/reports/confusion_matrix_<model>.csv
    outputs/reports/classification_metrics.json
"""

import os
import sys
import json
import joblib
import torch
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import get_project_root, setup_logger, load_jsonl

logger = setup_logger("evaluate_classifier")


def evaluate_tfidf_model():
    root = get_project_root()
    test_file = root / "data" / "processed" / "cuad_test.jsonl"
    model_path = root / "outputs" / "models" / "tfidf_baseline.joblib"
    reports_dir = root / "outputs" / "reports"

    logger.info(f"Loading TF-IDF baseline model from: {model_path}")
    pipeline = joblib.load(model_path)

    records = list(load_jsonl(test_file))
    X_test = [r["clause_text"] for r in records]
    y_test = [r["label"] for r in records]

    y_pred = pipeline.predict(X_test)
    labels = sorted(list(set(y_test)))

    acc = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_path = reports_dir / "confusion_matrix_tfidf.csv"
    cm_df.to_csv(cm_path)

    logger.info(f"TF-IDF Test Metrics: Acc={acc:.4f}, Macro F1={f1_macro:.4f}, Wtd F1={f1_wt:.4f}")
    logger.info(f"Confusion matrix saved to: {cm_path}")
    return {"accuracy": acc, "macro_f1": f1_macro, "weighted_f1": f1_wt}


def evaluate_legal_bert_model():
    root = get_project_root()
    test_file = root / "data" / "processed" / "cuad_test.jsonl"
    model_dir = root / "outputs" / "models" / "legal_bert_clause_classifier"
    reports_dir = root / "outputs" / "reports"

    if not model_dir.exists() or not (model_dir / "config.json").exists():
        logger.warning(f"Legal-BERT model checkpoint not found at {model_dir}")
        return None

    logger.info(f"Loading Legal-BERT model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    with open(model_dir / "id2label.json", "r", encoding="utf-8") as f:
        id2label = json.load(f)
        id2label = {int(k): v for k, v in id2label.items()}
        label2id = {v: k for k, v in id2label.items()}

    records = list(load_jsonl(test_file))
    X_test = [r["clause_text"] for r in records]
    y_test = [label2id[r["label"]] for r in records]

    preds = []
    batch_size = 32
    for i in range(0, len(X_test), batch_size):
        batch_texts = X_test[i:i + batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**enc)
            batch_preds = torch.argmax(outputs.logits, dim=1).tolist()
            preds.extend(batch_preds)

    acc = accuracy_score(y_test, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)

    target_names = [id2label[i] for i in range(len(id2label))]
    cm = confusion_matrix(y_test, preds)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_path = reports_dir / "confusion_matrix_legal_bert.csv"
    cm_df.to_csv(cm_path)

    logger.info(f"Legal-BERT Test Metrics: Acc={acc:.4f}, Macro F1={f1_macro:.4f}, Wtd F1={f1_wt:.4f}")
    logger.info(f"Confusion matrix saved to: {cm_path}")
    return {"accuracy": acc, "macro_f1": f1_macro, "weighted_f1": f1_wt}


if __name__ == "__main__":
    logger.info("Evaluating TF-IDF Baseline...")
    evaluate_tfidf_model()
    logger.info("\nChecking Legal-BERT evaluation...")
    evaluate_legal_bert_model()
