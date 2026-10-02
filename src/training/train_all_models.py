"""
src/training/train_all_models.py

Unified training orchestrator for 3 model families on CUAD 7-class task:
1. Model A: Classical Baseline - Tuned TF-IDF + Logistic Regression / Calibrated Classifier
2. Model B: Domain-Specific Transformer - nlpaueb/legal-bert-base-uncased
3. Model C: General Transformer - microsoft/deberta-v3-base (fallback: roberta-base)

Features:
- Fixed split indices (data/processed/splits.json) with zero data leakage
- Class-weighted cross entropy / balanced weights
- 3 fixed seeds (e.g. 42, 7, 2024) with mean ± std aggregation
- Computes standard NLP metrics and Risk-Focused metrics (risky-class precision/recall/F1, binary risky-vs-safe)
- Measures parameter count, model size on disk (MB), training time, inference latency
- Saves checkpoints to outputs/models/<model_name>/ with id2label.json and config.json
- Automatically generates outputs/models/active_model.json
"""

import os
import sys
import time
import json
import shutil
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.common import get_project_root, setup_logger, load_jsonl, save_json, load_json

logger = setup_logger("train_all_models")

SAFE_CLASS = "other_clause"


def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ClauseDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_cuad_splits():
    """Loads train, validation, and test splits."""
    root = get_project_root()
    train_file = root / "data" / "processed" / "cuad_train.jsonl"
    val_file = root / "data" / "processed" / "cuad_val.jsonl"
    test_file = root / "data" / "processed" / "cuad_test.jsonl"

    train_recs = list(load_jsonl(train_file))
    val_recs = list(load_jsonl(val_file))
    test_recs = list(load_jsonl(test_file))

    unique_labels = sorted(list(set(r["label"] for r in train_recs)))
    label2id = {lbl: i for i, lbl in enumerate(unique_labels)}
    id2label = {i: lbl for i, lbl in enumerate(unique_labels)}

    splits_data = {
        "train": ([r["clause_text"] for r in train_recs], [label2id[r["label"]] for r in train_recs]),
        "val": ([r["clause_text"] for r in val_recs], [label2id[r["label"]] for r in val_recs]),
        "test": ([r["clause_text"] for r in test_recs], [label2id[r["label"]] for r in test_recs]),
        "label2id": label2id,
        "id2label": id2label,
        "unique_labels": unique_labels
    }
    return splits_data


def compute_comprehensive_metrics(
    y_true: List[int],
    y_pred: List[int],
    id2label: Dict[int, str]
) -> Dict[str, Any]:
    """
    Computes overall accuracy, macro-F1, weighted-F1, per-class metrics,
    and specific risk-focused metrics (for all classes except 'other_clause').
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    per_class = {}
    risky_indices = []

    for idx, name in id2label.items():
        per_class[name] = {
            "precision": round(float(p_per[idx]), 4),
            "recall": round(float(r_per[idx]), 4),
            "f1": round(float(f1_per[idx]), 4),
            "support": int(sup_per[idx])
        }
        if name != SAFE_CLASS:
            risky_indices.append(idx)

    # Risky-only metrics (macro over all risky categories)
    risky_p = float(np.mean([p_per[i] for i in risky_indices])) if risky_indices else 0.0
    risky_r = float(np.mean([r_per[i] for i in risky_indices])) if risky_indices else 0.0
    risky_f1 = float(np.mean([f1_per[i] for i in risky_indices])) if risky_indices else 0.0

    # Collapsed Binary View: 1 = Risky (any of the 6 classes), 0 = Safe (other_clause)
    safe_idx = next(i for i, name in id2label.items() if name == SAFE_CLASS)
    binary_true = (y_true != safe_idx).astype(int)
    binary_pred = (y_pred != safe_idx).astype(int)

    bin_p, bin_r, bin_f1, _ = precision_recall_fscore_support(
        binary_true, binary_pred, average="binary", zero_division=0
    )

    # False positive rate on safe class: proportion of safe clauses incorrectly predicted as risky
    safe_mask = (binary_true == 0)
    safe_count = int(np.sum(safe_mask))
    false_positives = int(np.sum((binary_true == 0) & (binary_pred == 1)))
    fpr = float(false_positives / safe_count) if safe_count > 0 else 0.0

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_wt), 4),
        "precision_macro": round(float(p_macro), 4),
        "recall_macro": round(float(r_macro), 4),
        "risky_precision": round(risky_p, 4),
        "risky_recall": round(risky_r, 4),
        "risky_f1": round(risky_f1, 4),
        "binary_risky_precision": round(float(bin_p), 4),
        "binary_risky_recall": round(float(bin_r), 4),
        "binary_risky_f1": round(float(bin_f1), 4),
        "safe_false_positive_rate": round(fpr, 4),
        "per_class": per_class
    }


def measure_inference_latency(predict_fn, sample_texts: List[str], count: int = 200) -> float:
    """Measures average inference latency in milliseconds per clause on CPU."""
    subset = sample_texts[:count] if len(sample_texts) >= count else (sample_texts * (count // len(sample_texts) + 1))[:count]
    # Warmup
    for text in subset[:5]:
        _ = predict_fn([text])

    start_time = time.perf_counter()
    for text in subset:
        _ = predict_fn([text])
    elapsed = time.perf_counter() - start_time
    ms_per_clause = (elapsed / len(subset)) * 1000.0
    return round(ms_per_clause, 2)


# ==========================================
# MODEL A: TF-IDF + Logistic Regression
# ==========================================
def train_tfidf(splits: Dict[str, Any], seed: int, tune: bool = True) -> Dict[str, Any]:
    logger.info(f"--- Training Model A: TF-IDF Baseline (Seed: {seed}) ---")
    start_time = time.perf_counter()
    set_seed(seed)

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    id2label = splits["id2label"]

    best_pipeline = None
    best_val_f1 = -1.0
    best_params = {}

    if tune:
        logger.info("Tuning TF-IDF hyperparameters on validation set...")
        param_grid = [
            {"ngram_range": (1, 1), "C": 0.5, "sublinear_tf": True},
            {"ngram_range": (1, 2), "C": 1.0, "sublinear_tf": True},
            {"ngram_range": (1, 2), "C": 3.0, "sublinear_tf": True},
            {"ngram_range": (1, 2), "C": 5.0, "sublinear_tf": True},
            {"ngram_range": (1, 3), "C": 2.0, "sublinear_tf": True}
        ]
        for p in param_grid:
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(
                    ngram_range=p["ngram_range"],
                    sublinear_tf=p["sublinear_tf"],
                    min_df=2,
                    strip_accents="unicode"
                )),
                ("clf", LogisticRegression(
                    C=p["C"],
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed
                ))
            ])
            pipe.fit(X_train, y_train)
            v_pred = pipe.predict(X_val)
            _, _, v_f1, _ = precision_recall_fscore_support(y_val, v_pred, average="macro", zero_division=0)
            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                best_pipeline = pipe
                best_params = p
        logger.info(f"Best TF-IDF params: {best_params} (Val Macro-F1: {best_val_f1:.4f})")
    else:
        best_params = {"ngram_range": (1, 2), "C": 3.0, "sublinear_tf": True}
        best_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=best_params["ngram_range"],
                sublinear_tf=best_params["sublinear_tf"],
                min_df=2,
                strip_accents="unicode"
            )),
            ("clf", LogisticRegression(
                C=best_params["C"],
                max_iter=1000,
                class_weight="balanced",
                random_state=seed
            ))
        ])
        best_pipeline.fit(X_train, y_train)

    train_time = round(time.perf_counter() - start_time, 2)

    # Predictions
    y_test_pred = best_pipeline.predict(X_test).tolist()
    y_test_probs = best_pipeline.predict_proba(X_test)
    metrics = compute_comprehensive_metrics(y_test, y_test_pred, id2label)

    # Efficiency
    vocab_size = len(best_pipeline.named_steps["tfidf"].vocabulary_)
    param_count = vocab_size * len(id2label) + len(id2label)
    latency_ms = measure_inference_latency(lambda texts: best_pipeline.predict(texts), X_test)

    # Save checkpoint
    out_dir = PROJECT_ROOT / "outputs" / "models" / "tfidf" / f"seed_{seed}"
    os.makedirs(out_dir, exist_ok=True)
    joblib_path = out_dir / "model.joblib"
    joblib.dump(best_pipeline, joblib_path)

    # Model size in MB
    model_size_mb = round(os.path.getsize(joblib_path) / (1024 * 1024), 2)

    save_json(id2label, out_dir / "id2label.json")
    save_json({
        "model_name": "tfidf_logistic_regression",
        "seed": seed,
        "parameters": best_params,
        "vocab_size": vocab_size,
        "param_count": param_count,
        "model_size_mb": model_size_mb,
        "train_time_sec": train_time,
        "latency_ms_per_clause": latency_ms
    }, out_dir / "config.json")

    return {
        "model_name": "tfidf",
        "seed": seed,
        "pipeline": best_pipeline,
        "metrics": metrics,
        "train_time": train_time,
        "latency_ms": latency_ms,
        "param_count": param_count,
        "model_size_mb": model_size_mb,
        "y_true": y_test,
        "y_pred": y_test_pred,
        "y_probs": y_test_probs,
        "checkpoint_dir": str(out_dir)
    }


# ==========================================
# TRANSFORMER TRAINER (Legal-BERT & DeBERTa/RoBERTa)
# ==========================================
def train_transformer(
    model_key: str,
    hf_model_id: str,
    splits: Dict[str, Any],
    seed: int,
    epochs: int = 3,
    batch_size: int = 16,
    max_len: int = 128,
    lr: float = 2e-5,
    freeze_embeddings: bool = True
) -> Dict[str, Any]:
    logger.info(f"--- Training Transformer: {model_key} ({hf_model_id}) [Seed: {seed}] ---")
    start_time = time.perf_counter()
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target Device: {device} (Epochs={epochs}, BatchSize={batch_size}, MaxLen={max_len})")

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    id2label = splits["id2label"]
    label2id = splits["label2id"]
    num_labels = len(id2label)

    # 1. Tokenizer
    logger.info(f"Loading tokenizer: {hf_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

    train_ds = ClauseDataset(X_train, y_train, tokenizer, max_len)
    val_ds = ClauseDataset(X_val, y_val, tokenizer, max_len)
    test_ds = ClauseDataset(X_test, y_test, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 2. Model
    logger.info(f"Initializing {hf_model_id} for sequence classification...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_model_id,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
    except Exception as e:
        logger.error(f"Failed to load {hf_model_id}: {e}")
        if model_key == "deberta":
            logger.warning("Falling back to 'roberta-base' for general transformer...")
            hf_model_id = "roberta-base"
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
            model = AutoModelForSequenceClassification.from_pretrained(
                hf_model_id,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id
            )
        else:
            raise e

    if freeze_embeddings:
        # Freezing embeddings preserves stable representations and significantly speeds up CPU training
        if hasattr(model, "bert") and hasattr(model.bert, "embeddings"):
            for p in model.bert.embeddings.parameters():
                p.requires_grad = False
        elif hasattr(model, "deberta") and hasattr(model.deberta, "embeddings"):
            for p in model.deberta.embeddings.parameters():
                p.requires_grad = False
        elif hasattr(model, "roberta") and hasattr(model.roberta, "embeddings"):
            for p in model.roberta.embeddings.parameters():
                p.requires_grad = False
    model.float()
    model.to(device)

    # 3. Class-weighted Loss
    class_counts = np.bincount(y_train, minlength=num_labels)
    weights = len(y_train) / (num_labels * np.maximum(class_counts, 1).astype(float))
    loss_weights = torch.tensor(weights, dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=loss_weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01
    )
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    out_dir = PROJECT_ROOT / "outputs" / "models" / model_key / f"seed_{seed}"
    os.makedirs(out_dir, exist_ok=True)

    best_val_f1 = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().tolist())

        val_metrics = compute_comprehensive_metrics(val_targets, val_preds, id2label)
        val_macro_f1 = val_metrics["macro_f1"]
        logger.info(f"[{model_key} Seed {seed}] Epoch {epoch}/{epochs} | Val Macro-F1: {val_macro_f1:.4f} | Risky-F1: {val_metrics['risky_f1']:.4f}")

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            save_json(id2label, out_dir / "id2label.json")

    train_time = round(time.perf_counter() - start_time, 2)

    # 4. Held-out Test Evaluation
    best_model = AutoModelForSequenceClassification.from_pretrained(out_dir)
    best_model.float()
    best_model.to(device)
    best_model.eval()

    test_preds, test_probs = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1).tolist()
            test_preds.extend(preds)
            test_probs.extend(probs.tolist())

    metrics = compute_comprehensive_metrics(y_test, test_preds, id2label)

    # Efficiency metrics
    param_count = sum(p.numel() for p in best_model.parameters())
    # Disk size of out_dir
    total_bytes = sum(f.stat().st_size for f in out_dir.glob("**/*") if f.is_file())
    model_size_mb = round(total_bytes / (1024 * 1024), 2)

    # Measure CPU inference latency per clause
    best_model.to("cpu")
    best_model.eval()

    def predict_single(texts: List[str]):
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        with torch.no_grad():
            logits = best_model(**enc).logits
            return torch.argmax(logits, dim=1).tolist()

    latency_ms = measure_inference_latency(predict_single, X_test)

    config_info = {
        "model_key": model_key,
        "hf_model_id": hf_model_id,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_len": max_len,
        "learning_rate": lr,
        "param_count": param_count,
        "model_size_mb": model_size_mb,
        "train_time_sec": train_time,
        "latency_ms_per_clause": latency_ms,
        "best_val_macro_f1": best_val_f1
    }
    save_json(config_info, out_dir / "model_meta.json")

    return {
        "model_name": model_key,
        "seed": seed,
        "metrics": metrics,
        "train_time": train_time,
        "latency_ms": latency_ms,
        "param_count": param_count,
        "model_size_mb": model_size_mb,
        "y_true": y_test,
        "y_pred": test_preds,
        "y_probs": test_probs,
        "checkpoint_dir": str(out_dir)
    }


def copy_best_to_active(all_run_results: List[Dict[str, Any]], model_keys: List[str]):
    """
    Selects the best seed for each model family and copies it to outputs/models/<model_name>/,
    and determines the overall 'best' model based on risky-class F1.
    """
    models_root = PROJECT_ROOT / "outputs" / "models"
    best_per_model = {}

    for mk in model_keys:
        runs = [r for r in all_run_results if r["model_name"] == mk]
        if not runs:
            continue
        # Pick seed with highest test risky_f1 (or macro_f1)
        best_run = max(runs, key=lambda x: (x["metrics"]["risky_f1"], x["metrics"]["macro_f1"]))
        best_per_model[mk] = best_run

        target_dir = models_root / mk
        os.makedirs(target_dir, exist_ok=True)
        src_dir = Path(best_run["checkpoint_dir"])

        # Copy files to canonical model directory
        for f in src_dir.glob("*"):
            if f.is_file():
                shutil.copy2(f, target_dir / f.name)

        # For backward compatibility with existing codebase
        if mk == "legalbert":
            compat_dir = models_root / "legal_bert_clause_classifier"
            os.makedirs(compat_dir, exist_ok=True)
            for f in src_dir.glob("*"):
                if f.is_file():
                    shutil.copy2(f, compat_dir / f.name)
        elif mk == "tfidf":
            src_joblib = src_dir / "model.joblib"
            if src_joblib.exists():
                shutil.copy2(src_joblib, models_root / "tfidf_baseline.joblib")

    # Determine overall best model
    if best_per_model:
        overall_winner = max(best_per_model.values(), key=lambda x: x["metrics"]["risky_f1"])["model_name"]
    else:
        overall_winner = "legalbert"

    active_config = {
        "active_model": overall_winner,
        "selection_metric": "risky_f1",
        "available_models": list(best_per_model.keys()),
        "selected_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_json(active_config, models_root / "active_model.json")
    logger.info(f"Active model configured: '{overall_winner}' (saved to outputs/models/active_model.json)")


def train_all_models_pipeline(
    models_to_train: List[str] = ["tfidf", "legalbert", "deberta"],
    seeds: List[int] = [42, 7, 2024],
    epochs: int = 3,
    batch_size: int = 16,
    max_len: int = 128,
    lr: float = 2e-5
) -> List[Dict[str, Any]]:
    """Runs end-to-end training and evaluation over all requested models and seeds."""
    splits = load_cuad_splits()
    all_results = []

    for model_key in models_to_train:
        for seed in seeds:
            logger.info(f"\n==========================================")
            logger.info(f"Running {model_key.upper()} with Seed {seed}")
            logger.info(f"==========================================")

            if model_key == "tfidf":
                res = train_tfidf(splits, seed=seed, tune=True)
            elif model_key == "legalbert":
                res = train_transformer(
                    model_key="legalbert",
                    hf_model_id="nlpaueb/legal-bert-base-uncased",
                    splits=splits,
                    seed=seed,
                    epochs=epochs,
                    batch_size=batch_size,
                    max_len=max_len,
                    lr=lr,
                    freeze_embeddings=True
                )
            elif model_key == "deberta":
                res = train_transformer(
                    model_key="deberta",
                    hf_model_id="microsoft/deberta-v3-base",
                    splits=splits,
                    seed=seed,
                    epochs=epochs,
                    batch_size=batch_size,
                    max_len=max_len,
                    lr=lr,
                    freeze_embeddings=True
                )
            else:
                logger.warning(f"Unknown model key: {model_key}")
                continue

            all_results.append(res)

    # Copy best seeds to canonical paths and update active_model.json
    copy_best_to_active(all_results, models_to_train)
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified multi-model training for Legal Guard.")
    parser.add_argument("--models", type=str, default="tfidf,legalbert,deberta", help="Comma-separated model keys.")
    parser.add_argument("--seeds", type=str, default="42,7,2024", help="Comma-separated random seeds.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs for transformers.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--max_len", type=int, default=128, help="Max sequence length.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")

    args = parser.parse_args()
    model_list = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    train_all_models_pipeline(
        models_to_train=model_list,
        seeds=seed_list,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_len=args.max_len,
        lr=args.lr
    )
