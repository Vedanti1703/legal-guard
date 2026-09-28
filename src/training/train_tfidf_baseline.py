"""
src/training/train_tfidf_baseline.py

TF-IDF + Logistic Regression Baseline Model for Legal Clause Classification.
- Trains on data/processed/cuad_train.jsonl
- Validates on data/processed/cuad_val.jsonl
- Tests on data/processed/cuad_test.jsonl
- Evaluates: Accuracy, Precision, Recall, Macro F1, Weighted F1
- Saves model: outputs/models/tfidf_baseline.joblib
- Updates: outputs/reports/model_comparison.csv
"""

import os
import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import get_project_root, setup_logger, load_jsonl

logger = setup_logger("train_tfidf_baseline")


def load_dataset_split(filepath: Path):
    records = list(load_jsonl(filepath))
    texts = [r["clause_text"] for r in records]
    labels = [r["label"] for r in records]
    return texts, labels


def train_tfidf_baseline():
    root = get_project_root()
    train_file = root / "data" / "processed" / "cuad_train.jsonl"
    val_file = root / "data" / "processed" / "cuad_val.jsonl"
    test_file = root / "data" / "processed" / "cuad_test.jsonl"

    models_dir = root / "outputs" / "models"
    reports_dir = root / "outputs" / "reports"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    logger.info("Loading train, val, and test splits (document-split, zero leakage)...")
    X_train, y_train = load_dataset_split(train_file)
    X_val, y_val = load_dataset_split(val_file)
    X_test, y_test = load_dataset_split(test_file)

    logger.info(f"Dataset sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Build Pipeline: TF-IDF with sublinear scaling and n-grams + Balanced Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=15000,
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode"
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    logger.info("Training TF-IDF + Logistic Regression baseline...")
    pipeline.fit(X_train, y_train)

    # Evaluate on Validation set
    y_val_pred = pipeline.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_p_macro, val_r_macro, val_f1_macro, _ = precision_recall_fscore_support(y_val, y_val_pred, average="macro", zero_division=0)
    val_p_wt, val_r_wt, val_f1_wt, _ = precision_recall_fscore_support(y_val, y_val_pred, average="weighted", zero_division=0)

    logger.info(f"Validation Results: Acc={val_acc:.4f}, Macro F1={val_f1_macro:.4f}, Weighted F1={val_f1_wt:.4f}")

    # Evaluate on Test set
    y_test_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_p_macro, test_r_macro, test_f1_macro, _ = precision_recall_fscore_support(y_test, y_test_pred, average="macro", zero_division=0)
    test_p_wt, test_r_wt, test_f1_wt, _ = precision_recall_fscore_support(y_test, y_test_pred, average="weighted", zero_division=0)

    logger.info(f"Test Results: Acc={test_acc:.4f}, Macro F1={test_f1_macro:.4f}, Weighted F1={test_f1_wt:.4f}")

    report_str = classification_report(y_test, y_test_pred, zero_division=0)
    logger.info(f"\nTest Classification Report:\n{report_str}")

    # Save model
    model_save_path = models_dir / "tfidf_baseline.joblib"
    joblib.dump(pipeline, model_save_path)
    logger.info(f"Model saved to: {model_save_path}")

    # Save metrics to model_comparison.csv
    comparison_file = reports_dir / "model_comparison.csv"
    baseline_metrics = {
        "model": "TF-IDF + Logistic Regression",
        "split": "test",
        "accuracy": round(test_acc, 4),
        "precision_macro": round(test_p_macro, 4),
        "recall_macro": round(test_r_macro, 4),
        "macro_f1": round(test_f1_macro, 4),
        "weighted_f1": round(test_f1_wt, 4)
    }

    if comparison_file.exists():
        comp_df = pd.read_csv(comparison_file)
        # Remove previous row if exists
        comp_df = comp_df[comp_df["model"] != baseline_metrics["model"]]
        comp_df = pd.concat([comp_df, pd.DataFrame([baseline_metrics])], ignore_index=True)
    else:
        comp_df = pd.DataFrame([baseline_metrics])

    comp_df.to_csv(comparison_file, index=False)
    logger.info(f"Updated model comparison table: {comparison_file}")

    # Save detailed classification report
    clf_report_path = reports_dir / "baseline_classification_report.txt"
    with open(clf_report_path, "w", encoding="utf-8") as f:
        f.write("TF-IDF + LOGISTIC REGRESSION BASELINE REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Accuracy:    {test_acc:.4f}\n")
        f.write(f"Macro F1:    {test_f1_macro:.4f}\n")
        f.write(f"Weighted F1: {test_f1_wt:.4f}\n\n")
        f.write(report_str)

    return baseline_metrics


if __name__ == "__main__":
    train_tfidf_baseline()
