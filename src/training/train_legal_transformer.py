"""
src/training/train_legal_transformer.py

Fine-tunes a Legal-BERT transformer (nlpaueb/legal-bert-base-uncased)
for multi-class legal clause classification across 7 categories:
1. Liquidated Damages
2. Notice Period To Terminate Renewal
3. Post-Termination Services
4. Renewal Term
5. Revenue/Profit Sharing
6. Termination For Convenience
7. other_clause (background / non-target clauses)

Design & Efficiency Highlights:
- Strict document-level train/validation/test splits (zero leakage)
- Class-weighted Cross-Entropy Loss to counter label imbalance
- Parameter freezing of lower BERT layers for fast, CPU-friendly training
- Max sequence length = 128 (matches median legal clause length ~31 words)
- Saves checkpoints to: outputs/models/legal_bert_clause_classifier
- Updates: outputs/reports/model_comparison.csv
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import get_project_root, setup_logger, load_jsonl

logger = setup_logger("train_legal_transformer")

MODEL_CHECKPOINT = "nlpaueb/legal-bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
SEED = 42


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LegalClauseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
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
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return item


def load_data(filepath: Path, label2id: dict):
    records = list(load_jsonl(filepath))
    texts = [r["clause_text"] for r in records]
    labels = [label2id[r["label"]] for r in records]
    return texts, labels


def compute_class_weights(labels_list: list, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels_list, minlength=num_classes)
    total = len(labels_list)
    # Inverse frequency weighting with smoothing
    weights = total / (num_classes * np.maximum(counts, 1).astype(float))
    weights = torch.tensor(weights, dtype=torch.float)
    return weights


def train_legal_transformer():
    set_seed(SEED)
    root = get_project_root()
    train_file = root / "data" / "processed" / "cuad_train.jsonl"
    val_file = root / "data" / "processed" / "cuad_val.jsonl"
    test_file = root / "data" / "processed" / "cuad_test.jsonl"

    output_model_dir = root / "outputs" / "models" / "legal_bert_clause_classifier"
    reports_dir = root / "outputs" / "reports"
    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Determine labels
    all_train_records = list(load_jsonl(train_file))
    unique_labels = sorted(list(set(r["label"] for r in all_train_records)))
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for i, l in enumerate(unique_labels)}
    num_labels = len(unique_labels)

    logger.info(f"Unique categories ({num_labels}): {unique_labels}")

    # 2. Tokenizer & Dataset
    logger.info(f"Loading Legal-BERT tokenizer: {MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    train_texts, train_labels = load_data(train_file, label2id)
    val_texts, val_labels = load_data(val_file, label2id)
    test_texts, test_labels = load_data(test_file, label2id)

    train_dataset = LegalClauseDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = LegalClauseDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    test_dataset = LegalClauseDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device}")

    logger.info(f"Loading pre-trained Legal-BERT model: {MODEL_CHECKPOINT}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # Freeze embeddings and lower 8 encoder layers for fast CPU training
    # Fine-tune the top 4 layers (8-11) + pooler + classifier head
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for layer in model.bert.encoder.layer[:8]:
        for param in layer.parameters():
            param.requires_grad = False

    model.to(device)

    # 4. Class weighting & Loss
    class_weights = compute_class_weights(train_labels, num_labels).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    logger.info(f"Starting Training: {EPOCHS} epochs, Batch size={BATCH_SIZE}, Steps={total_steps}")

    best_val_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_targets, val_preds)
        _, _, val_f1_macro, _ = precision_recall_fscore_support(val_targets, val_preds, average="macro", zero_division=0)
        _, _, val_f1_wt, _ = precision_recall_fscore_support(val_targets, val_preds, average="weighted", zero_division=0)

        logger.info(f"Epoch {epoch}/{EPOCHS} -> Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1_macro:.4f} | Val Wtd F1: {val_f1_wt:.4f}")

        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            logger.info(f"New best validation Macro F1 ({best_val_f1:.4f}). Saving model checkpoint...")
            model.save_pretrained(output_model_dir)
            tokenizer.save_pretrained(output_model_dir)
            # Save label mappings
            with open(output_model_dir / "id2label.json", "w", encoding="utf-8") as f:
                json.dump(id2label, f, indent=2)

    logger.info("Training complete. Evaluating best checkpoint on held-out Test set...")
    # Load best model for evaluation
    best_model = AutoModelForSequenceClassification.from_pretrained(output_model_dir)
    best_model.to(device)
    best_model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_targets, test_preds)
    test_p_macro, test_r_macro, test_f1_macro, _ = precision_recall_fscore_support(test_targets, test_preds, average="macro", zero_division=0)
    test_p_wt, test_r_wt, test_f1_wt, _ = precision_recall_fscore_support(test_targets, test_preds, average="weighted", zero_division=0)

    target_names = [id2label[i] for i in range(num_labels)]
    report_str = classification_report(test_targets, test_preds, target_names=target_names, zero_division=0)
    logger.info(f"\nFinal Legal-BERT Test Classification Report:\n{report_str}")

    # Save detailed report
    report_path = reports_dir / "legal_bert_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("LEGAL-BERT (nlpaueb/legal-bert-base-uncased) CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Test Accuracy:    {test_acc:.4f}\n")
        f.write(f"Test Macro F1:    {test_f1_macro:.4f}\n")
        f.write(f"Test Weighted F1: {test_f1_wt:.4f}\n\n")
        f.write(report_str)

    # Update model_comparison.csv
    comparison_file = reports_dir / "model_comparison.csv"
    bert_metrics = {
        "model": "Fine-Tuned Legal-BERT",
        "split": "test",
        "accuracy": round(test_acc, 4),
        "precision_macro": round(test_p_macro, 4),
        "recall_macro": round(test_r_macro, 4),
        "macro_f1": round(test_f1_macro, 4),
        "weighted_f1": round(test_f1_wt, 4)
    }

    if comparison_file.exists():
        comp_df = pd.read_csv(comparison_file)
        comp_df = comp_df[comp_df["model"] != bert_metrics["model"]]
        comp_df = pd.concat([comp_df, pd.DataFrame([bert_metrics])], ignore_index=True)
    else:
        comp_df = pd.DataFrame([bert_metrics])

    comp_df.to_csv(comparison_file, index=False)
    logger.info(f"Updated model comparison table: {comparison_file}")
    return bert_metrics


if __name__ == "__main__":
    train_legal_transformer()
