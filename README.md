# NLP for Detecting Hidden Fees and Unfavorable Terms in Consumer Terms & Conditions

An academic and practical Natural Language Processing pipeline for contract understanding, semantic clause retrieval, and Indian consumer law domain evaluation.

---

## 1. Project Overview & Architecture

```
Consumer T&C / Contract
          │
          ▼
   PDF / Text Extraction (PyMuPDF)
          │
          ▼
  Clause Segmentation & Normalization (Legal Markers & Patterns)
          │
          ├─────────────────────────────────────────────────┐
          │                                                 │
          ▼                                                 ▼
[CUAD Multi-Class Classifier]                     [ACORD Semantic Retrieval]
(Legal-BERT / TF-IDF Baseline)                   (Sentence-BERT Dense Search)
Identifies clause categories:                    Finds matching unfavorable terms
- Renewal Term                                   e.g. "Find clauses related to
- Termination for Convenience                    automatic renewal or penalty"
- Liquidated Damages, etc.
          │                                                 │
          └───────────────────────┬─────────────────────────┘
                                  │
                                  ▼
                   [Indian Legal Domain Grounding]
        Evaluates risk under Indian Contract Act 1872 &
              Curated Risky Pattern Knowledge Base
                                  │
                                  ▼
                [Information Extraction Pipeline]
              Amounts, Triggers, Deadlines, Remedies
```

---

## 2. Directory Structure

```
nlp/
├── data/
│   ├── raw/                  # Extracted raw files from CUAD, ACORD, Indian datasets
│   ├── processed/            # Cleaned, normalized JSONL and label mapping files
│   └── custom/               # Project annotation template for consumer T&Cs
├── outputs/
│   ├── reports/              # Inspection reports, data quality checks, model metrics
│   └── models/               # Saved baseline, transformer, and embedding checkpoints
├── src/
│   ├── preprocessing/        # Inspection, extraction, normalization, and split scripts
│   ├── training/             # TF-IDF baseline, Legal-BERT classifier, Dense Retriever
│   ├── evaluation/           # Classifier metrics, ACORD IR metrics, Indian evaluation
│   └── utils/                # Paths, text cleaning, shared helpers
├── requirements.txt
└── README.md
```

---

## 3. Dataset Roles

| Dataset | Primary Function | Total Scope | Annotation Type |
|---|---|---|---|
| **CUAD** | Clause Understanding & Multi-Class Classification | 510 contracts, 13,823 labeled spans | 41 legal categories (span-level answers) |
| **ACORD** | Dense Semantic Clause Retrieval | 114 queries, 3,931 corpus clauses | 126,659 query-clause pairs with 0–4 relevance ratings |
| **Indian Legal Dataset** | Indian Law Domain Grounding & Risky Pattern Reference | 5 PDFs (136 pages) | Statutory provisions (ICA 1872) & 12 risky clause patterns |

---

## 4. Setup & Running Inspection

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run inspection scripts
python src/preprocessing/inspect_cuad.py
python src/preprocessing/inspect_acord.py
python src/preprocessing/inspect_indian.py

# 3. Generate master inspection report
python src/preprocessing/generate_inspection_report.py
```
