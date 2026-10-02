#!/usr/bin/env bash
# =========================================================================
# run_all.sh - Full Pipeline Execution for Legal Guard NLP
# =========================================================================
set -e

echo "[1/5] Training 3 Model Families (TF-IDF, Legal-BERT, DeBERTa-v3) across 3 seeds..."
python src/training/train_all_models.py --models tfidf,legalbert,deberta --seeds 42,7,2024 --epochs 2 --batch_size 32 --max_len 128

echo ""
echo "[2/5] Running Comparative Analysis and Generating Figures..."
python src/evaluation/compare_models.py

echo ""
echo "[3/5] Tuning Two-Stage Risk Gating Filter and Evaluating False-Positive Drop..."
python src/evaluation/tune_risk_filter.py --recall_floor 0.70

echo ""
echo "[4/5] Evaluating Whole-Document Summarizer (Extractive, Abstractive, Hybrid)..."
python src/evaluation/evaluate_summarizer.py

echo ""
echo "[5/5] Running Pytest Suite..."
python -m pytest tests/ -v

echo ""
echo "========================================================================="
echo "All pipeline stages completed successfully!"
echo "Starting Web Server on http://127.0.0.1:5000 ..."
echo "========================================================================="
python server.py
