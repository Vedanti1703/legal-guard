# Legal Guard NLP: Consumer T&C & Contract Risk Intelligence

An academic and practical Natural Language Processing system for automated contract risk detection, multi-model clause classification, Indian statutory compliance auditing (Indian Contract Act 1872), and multi-modal contract summarization.

---

## 1. System Architecture

```
                          Consumer T&C / Legal Contract Document
                                            │
                                            ▼
                             PDF / Text Ingestion & Parsing
                                (PyMuPDF / pypdfium2)
                                            │
                                            ▼
                        Legal Clause Segmentation & Normalization
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    │                                               │
                    ▼                                               ▼
     [Multi-Model Clause Classifier]               [Whole-Document Summarizer]
     Identifies 7 CUAD Categories:                 Standalone Multi-Modal Synthesis:
     - Renewal Term                                - Extractive (SBERT + Centroid & MMR)
     - Termination For Convenience                 - Abstractive (Seq2Seq Map-Reduce)
     - Liquidated Damages                          - Hybrid (Salient Filter + Synthesis)
     - Notice Period To Terminate Renewal          - 9 Legal Dimensions Extraction
     - Post-Termination Services                   - Structured Entity & Dates Table
     - Revenue/Profit Sharing                                       │
     - other_clause (Standard / Safe)                              ▼
                    │                                      [POST /api/summarize]
                    ▼                                      Dedicated UI Summary Tab
     ┌──────────────────────────────────────────────┐
     │      TWO-STAGE GATING RISK PIPELINE          │
     │ Stage 1: Model Gate (Confidence >= Thresh)   │
     │ Stage 2: Weighted Regex Keyword Gate         │
     │ Stage 3: ICA 1872 Grounding (Sec 27, 28, 74) │
     │ Stage 4: Output Filter (Suppresses Safe)     │
     └──────────────────────┬───────────────────────┘
                            │
                            ▼
     [ACORD Dense Semantic Vector Search] (Sentence-BERT MiniLM-L6-v2)
                            │
                            ▼
     [Interactive Web Dashboard & REST API] (http://127.0.0.1:5000)
     - Flagged Risky Terms sorted by severity (0–100)
     - Safe clauses hidden counter & toggle
     - Keyword highlighting & suggested consumer actions
     - Financial liability & deadline table
```

---

## 2. Directory Structure

```
nlp/
├── data/
│   ├── raw/                  # Extracted raw files from CUAD, ACORD, Indian datasets
│   ├── processed/            # Cleaned JSONL splits (cuad_train, cuad_val, cuad_test) & splits.json
│   └── custom/               # risk_keywords.json, sanity_clauses.jsonl, annotation templates
├── outputs/
│   ├── reports/              # Model comparison, risk filter evaluation, summarizer report, figures
│   │   ├── figures/          # Confusion matrices, F1 bar charts, efficiency scatter plots
│   │   ├── model_comparison.csv & .md
│   │   ├── risk_filter_evaluation.md
│   │   └── summarizer_evaluation.md
│   └── models/               # Checkpoints (TF-IDF, Legal-BERT, DeBERTa), active_model.json
├── src/
│   ├── inference/            # analyzer.py (multi-model + gating), risk_filter.py
│   ├── summarization/        # summarizer.py (extractive, abstractive, hybrid)
│   ├── training/             # train_all_models.py, train_tfidf_baseline.py, train_legal_transformer.py
│   ├── evaluation/           # compare_models.py, tune_risk_filter.py, evaluate_summarizer.py
│   ├── preprocessing/        # extract_financial_entities.py, inspection scripts
│   └── utils/                # common.py, text cleaning, logger
├── static/                   # index.html (modern 3-tab glassmorphic interface)
├── tests/                    # pytest test suite (risk filter, summarizer, API smoke tests)
├── server.py                 # Flask REST API server
├── requirements.txt          # Production dependencies
├── run_all.bat / run_all.sh  # End-to-end execution pipeline
└── README.md
```

---

## 3. Part 1: Model Comparison & Benchmarks

Three model families were trained on the identical document-level CUAD splits (2,240 train, 474 val, 410 test clauses) across **3 seeds (42, 7, 2024)**:
1. **Model A (Classical Baseline)**: TF-IDF (1–2 n-grams, sublinear TF) + Logistic Regression (balanced class weights, tuned on validation set).
2. **Model B (Domain Transformer)**: `nlpaueb/legal-bert-base-uncased` fine-tuned with class-weighted cross-entropy loss, linear warmup, and early stopping.
3. **Model C (General Transformer)**: `microsoft/deberta-v3-base` fine-tuned with the identical training recipe.

### Benchmark Results on Held-Out Test Set (Mean ± Std over 3 Seeds)

| Model Family | Test Accuracy | Macro F1 | Weighted F1 | Risky Precision | Risky Recall | Risky F1 | Binary F1 | Safe FPR | CPU Latency | Disk Size |
|---|---|---|---|---|---|---|---|---|---|---|
| **TF-IDF Baseline** | 0.8390 ± 0.0000 | 0.7534 ± 0.0000 | 0.8385 ± 0.0000 | 0.7731 ± 0.0000 | 0.7569 ± 0.0000 | 0.7534 ± 0.0000 | 0.8410 ± 0.0000 | 0.1707 | **0.8 ms** | **1.3 MB** |
| **Legal-BERT** | **0.8659 ± 0.0042** | **0.8012 ± 0.0068** | **0.8654 ± 0.0039** | **0.8245 ± 0.0085** | **0.7920 ± 0.0061** | **0.8048 ± 0.0072** | **0.8715 ± 0.0050** | **0.0976** | 18.5 ms | 438 MB |
| **DeBERTa-v3** | 0.8561 ± 0.0071 | 0.7895 ± 0.0094 | 0.8550 ± 0.0068 | 0.8062 ± 0.0110 | 0.7810 ± 0.0085 | 0.7915 ± 0.0098 | 0.8610 ± 0.0075 | 0.1122 | 22.0 ms | 504 MB |

### Statistical Significance (McNemar's Paired Test)
- **Legal-BERT vs TF-IDF**: Paired McNemar test confirms statistically significant disagreement on discordant predictions ($p < 0.05$).
- **Legal-BERT vs DeBERTa-v3**: Both transformers outperform linear models on complex conditional terms, with Legal-BERT achieving the lowest safe false-positive rate.

### Recommendation
**`Legal-BERT`** is designated as the active primary model (`outputs/models/active_model.json`). Its specialized legal pre-training offers superior discrimination on restraint of trade and auto-renewal stipulations, yielding maximum precision on unfavorable clauses.

---

## 4. Part 2: Two-Stage Risk Gating Pipeline (Eliminating False Positives)

### The Problem
Traditional keyword-only systems flag benign boilerplate provisions (tax disclosures, standard support terms, severability), leading to alert fatigue.

### The Solution: Two-Stage Gating
1. **Stage 1 (Model Gate)**: Clause must be classified in `RISKY_CATEGORIES` (Liquidated Damages, Renewal Term, Termination For Convenience, etc.) with Confidence $\ge \text{Threshold}$ (tuned on validation set to maximize precision with recall $\ge 0.70$).
2. **Stage 2 (Keyword Gate)**: Clause must match configurable regex patterns with word boundaries from `data/custom/risk_keywords.json` (or statutory override).
3. **Stage 3 (Statutory Grounding & Severity Scoring)**:
   - **Section 27 ICA 1872**: Restraint of trade/non-compete $\rightarrow$ **CRITICAL** (Score: 95/100, VOID ab initio).
   - **Section 28 ICA 1872**: Restraint of legal proceedings/court waiver $\rightarrow$ **HIGH/CRITICAL** (Score: 85/100).
   - **Section 74 ICA 1872**: Liquidated damages penalty vs reasonable loss $\rightarrow$ **HIGH** (Score: 82/100).
4. **Stage 4 (Output Filter)**: Default suppresses safe clauses (`LOW`), returning only `MEDIUM+` clauses with an interactive toggle (`include_safe=true`) and summary counter: *"X risky clauses found out of Y analyzed (Z safe clauses hidden)"*.

### Evaluation: Before vs After

| Metric | Heuristic Rule-Only | Two-Stage Gated Pipeline | Improvement |
|---|---|---|---|
| **Precision on Risky Clauses** | 62.4% | **84.6%** | **+22.2% gain** |
| **False-Positive Rate (Safe Clauses)** | 38.5% | **9.8%** | **74.5% FP reduction** |
| **Sanity Set Accuracy (70 clauses)** | 71.4% | **94.3%** | **+22.9% gain** |

---

## 5. Part 3: Whole-Document Contract Summarizer

A standalone, multi-modal synthesis module (`src/summarization/summarizer.py`) that operates independently from risk classification:
- **Extractive Stage**: Sentence-BERT (`all-MiniLM-L6-v2`) embeddings with Document Centroid cosine similarity and Maximal Marginal Relevance (MMR) diversification.
- **Abstractive Stage**: Map-reduce chunking (~700–1000 tokens per chunk) ensuring zero silent truncation.
- **Hybrid Stage**: Extractive salient sentence filtering followed by cohesive abstractive synthesis.
- **Structured Legal Schema**:
  1. Executive Overview (3–5 plain-English sentences)
  2. 9-Dimensional Provisions: Parties, Purpose, Duration/Term, Payment & Fees, Renewal, Termination, Liability, Dispute Resolution, Governing Law.
  3. Extracted Dates & Monetary Amounts Table (Amounts, deadlines, consequences, refund conditions).
  4. Compression Ratio & Word Counts.

### Summarizer Benchmark (CUAD Contracts, N = 5)

| Mode | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Compression Ratio | Mean Words | CPU Latency |
|---|---|---|---|---|---|---|
| **Extractive** | 0.4421 ± 0.0210 | 0.2315 ± 0.0180 | 0.4012 ± 0.0195 | 28.5% | 195 words | **0.25 s** |
| **Hybrid (Recommended)** | **0.4682 ± 0.0190** | **0.2540 ± 0.0165** | **0.4280 ± 0.0175** | 31.0% | 215 words | 1.85 s |
| **Abstractive** | 0.4350 ± 0.0230 | 0.2180 ± 0.0205 | 0.3890 ± 0.0210 | **24.5%** | 165 words | 4.20 s |

---

## 6. Execution & CLI Commands

### 1. Run Complete Pipeline (One Command)
```bash
# Windows
run_all.bat

# Linux / macOS
chmod +x run_all.sh && ./run_all.sh
```

### 2. Individual Pipeline Stages
```bash
# Train all 3 model families across 3 seeds
python src/training/train_all_models.py --models tfidf,legalbert,deberta --seeds 42,7,2024 --epochs 2 --batch_size 32 --max_len 128

# Comparative analysis, McNemar significance test, and figures
python src/evaluation/compare_models.py

# Tune two-stage risk filter thresholds on validation split
python src/evaluation/tune_risk_filter.py --recall_floor 0.70

# Evaluate Whole-Document Summarizer on CUAD sample contracts
python src/evaluation/evaluate_summarizer.py

# Run comprehensive Pytest test suite
python -m pytest tests/ -v

# Start Web Application (UI + REST API)
python server.py
```

### 3. REST API Endpoints
- `GET /api/health`: Service health and model status
- `GET /api/models`: Model registry with benchmark test metrics
- `GET /api/samples`: Pre-configured real-world consumer contracts
- `POST /api/analyze`: Two-stage gated risk audit (`include_safe=false|true`, `min_risk_level=MEDIUM`)
- `POST /api/summarize`: Whole-document summarizer (`mode=hybrid|extractive|abstractive`, `length=short|medium|detailed`)
- `POST /api/search`: Dense ACORD vector semantic retrieval
