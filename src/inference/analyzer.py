"""
src/inference/analyzer.py

Unified inference & analysis pipeline for Consumer T&C / Contract NLP:
1. Legal Clause Segmentation & Normalization
2. Legal-BERT Multi-Class Classifier + TF-IDF Baseline Comparison
3. Rule-based Financial Entity & Consequence Extraction
4. Indian Law Grounding (ICA 1872 Sec 27, 28, 73/74) & Consumer Risk Scoring
5. Dense Vector Semantic Clause Retrieval (Sentence-BERT / ACORD)
"""

import os
import sys
import re
import json
import logging
import torch
import joblib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.common import clean_legal_text, setup_logger, load_json
from src.preprocessing.extract_financial_entities import extract_financial_terms

logger = setup_logger("inference_analyzer")

# Global Cache for Models
_MODELS_CACHE = {}

def get_models():
    """Lazy loads and caches classification and embedding models."""
    if _MODELS_CACHE.get("loaded"):
        return _MODELS_CACHE

    logger.info("Initializing NLP models for inference...")
    
    # 1. Load TF-IDF Baseline
    tfidf_path = PROJECT_ROOT / "outputs" / "models" / "tfidf_baseline.joblib"
    if tfidf_path.exists():
        _MODELS_CACHE["tfidf"] = joblib.load(tfidf_path)
        logger.info("Loaded TF-IDF model.")
    else:
        _MODELS_CACHE["tfidf"] = None
        logger.warning("TF-IDF model file not found.")

    # 2. Load Legal-BERT Transformer
    bert_dir = PROJECT_ROOT / "outputs" / "models" / "legal_bert_clause_classifier"
    if bert_dir.exists():
        try:
            tokenizer = AutoTokenizer.from_pretrained(bert_dir)
            model = AutoModelForSequenceClassification.from_pretrained(bert_dir)
            model.eval()
            _MODELS_CACHE["bert_tokenizer"] = tokenizer
            _MODELS_CACHE["bert_model"] = model
            
            # Load id2label mapping
            id2label_path = bert_dir / "id2label.json"
            if id2label_path.exists():
                id2label = load_json(id2label_path)
                # Convert string keys to int if necessary
                _MODELS_CACHE["id2label"] = {int(k): v for k, v in id2label.items()}
            else:
                _MODELS_CACHE["id2label"] = {
                    0: "Liquidated Damages",
                    1: "Notice Period To Terminate Renewal",
                    2: "Post-Termination Services",
                    3: "Renewal Term",
                    4: "Revenue/Profit Sharing",
                    5: "Termination For Convenience",
                    6: "other_clause"
                }
            logger.info("Loaded Legal-BERT transformer model.")
        except Exception as e:
            logger.error(f"Failed to load Legal-BERT: {e}")
            _MODELS_CACHE["bert_model"] = None
    else:
        _MODELS_CACHE["bert_model"] = None

    # 3. Load Sentence Transformer for Semantic Search
    try:
        sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        _MODELS_CACHE["sbert"] = sbert
        logger.info("Loaded Sentence-BERT model for retrieval.")
    except Exception as e:
        logger.error(f"Failed to load Sentence-BERT: {e}")
        _MODELS_CACHE["sbert"] = None

    # 4. Load ACORD Pre-computed Vector Index if available
    acord_emb_path = PROJECT_ROOT / "outputs" / "models" / "acord_corpus_embeddings.pt"
    acord_meta_path = PROJECT_ROOT / "outputs" / "models" / "acord_corpus_metadata.json"
    if acord_emb_path.exists() and acord_meta_path.exists():
        try:
            _MODELS_CACHE["acord_embeddings"] = torch.load(acord_emb_path)
            _MODELS_CACHE["acord_metadata"] = load_json(acord_meta_path)
            logger.info("Loaded pre-computed ACORD vector index.")
        except Exception as e:
            logger.warning(f"Could not load ACORD vector index: {e}")

    # Unified Category Map
    label_map_path = PROJECT_ROOT / "data" / "processed" / "label_mapping.json"
    if label_map_path.exists():
        _MODELS_CACHE["cuad_to_unified"] = load_json(label_map_path).get("cuad_to_unified", {})
    else:
        _MODELS_CACHE["cuad_to_unified"] = {
            "Renewal Term": "automatic_renewal",
            "Notice Period To Terminate Renewal": "renewal_notice",
            "Termination For Convenience": "termination",
            "Liquidated Damages": "penalty",
            "Post-Termination Services": "post_termination",
            "Revenue/Profit Sharing": "revenue_sharing",
            "Price Restrictions": "price_increase_condition",
            "Cap On Liability": "liability_limitation"
        }

    _MODELS_CACHE["loaded"] = True
    return _MODELS_CACHE


def segment_clauses(full_text: str) -> List[Dict[str, Any]]:
    """
    Intelligently splits a contract/T&C document into discrete, numbered clauses.
    """
    cleaned = clean_legal_text(full_text)
    if not cleaned:
        return []

    # Try matching explicit legal section headers: e.g. "1.", "1.1", "Section 2", "Clause 4", "Article 5", "SECTION 1:"
    pattern = r'\n(?=(?:[0-9]{1,2}\.[0-9]{0,2}\b|SECTION\s+[0-9]{1,2}|Clause\s+[0-9]{1,2}|Article\s+[0-9]{1,2}|\[?[A-Z0-9]{1,3}\]\s+|\([0-9a-z]\)\s+|[0-9]{1,2}\.\s+))'
    splits = re.split(pattern, cleaned, flags=re.IGNORECASE)

    # Fallback to paragraph splitting if no numbered sections found
    if len(splits) <= 1:
        splits = [p.strip() for p in cleaned.split("\n\n") if p.strip()]

    clauses = []
    for idx, text_block in enumerate(splits, 1):
        text_block = text_block.strip()
        if not text_block or len(text_block) < 15:
            continue

        # Extract title if present (e.g. "1. Automatic Renewal. The subscription...")
        lines = text_block.splitlines()
        first_line = lines[0].strip()
        header = f"Clause {idx}"
        if len(first_line) < 80 and any(kw in first_line.lower() for kw in ["section", "clause", "article", ".", ":", "term", "fee", "cancellation", "payment", "liability", "refund", "renewal"]):
            header = first_line[:60]

        clauses.append({
            "clause_id": f"C-{idx:03d}",
            "title": header,
            "text": text_block
        })

    return clauses


def classify_clause(clause_text: str) -> Dict[str, Any]:
    """
    Predicts clause category using Legal-BERT and TF-IDF models.
    """
    cache = get_models()
    cuad_to_unified = cache.get("cuad_to_unified", {})

    bert_pred = "other_clause"
    bert_conf = 0.0
    bert_probs_dict = {}

    # 1. Legal-BERT prediction
    bert_model = cache.get("bert_model")
    tokenizer = cache.get("bert_tokenizer")
    id2label = cache.get("id2label", {})

    if bert_model and tokenizer:
        inputs = tokenizer(clause_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            top_idx = torch.argmax(probs).item()
            bert_pred = id2label.get(top_idx, "other_clause")
            bert_conf = float(probs[top_idx].item())

            for i, p in enumerate(probs):
                lbl = id2label.get(i, f"class_{i}")
                bert_probs_dict[lbl] = round(float(p.item()), 4)

    # 2. TF-IDF prediction
    tfidf_model = cache.get("tfidf")
    tfidf_pred = "other_clause"
    tfidf_conf = 0.0

    if tfidf_model:
        try:
            preds = tfidf_model.predict([clause_text])[0]
            probs = tfidf_model.predict_proba([clause_text])[0]
            tfidf_pred = str(preds)
            tfidf_conf = float(np.max(probs))
        except Exception:
            pass

    unified_category = cuad_to_unified.get(bert_pred, bert_pred)

    return {
        "legal_bert_category": bert_pred,
        "legal_bert_confidence": round(bert_conf, 4),
        "legal_bert_probabilities": bert_probs_dict,
        "tfidf_category": tfidf_pred,
        "tfidf_confidence": round(tfidf_conf, 4),
        "unified_category": unified_category
    }


def evaluate_risk_and_indian_law(clause_text: str, category: str, entities: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates risks under Indian Contract Act 1872 & Consumer Protection standards.
    """
    text_lower = clause_text.lower()
    risk_reasons = []
    highlights = []
    risk_level = "LOW"
    risk_score = 10  # 0 to 100
    statutory_ref = "N/A"

    # --- RULE 1: Restraint of Trade / Profession (Section 27 Indian Contract Act 1872) ---
    if any(k in text_lower for k in ["shall not engage", "restraint of trade", "non-compete", "prohibited from working", "competing business for a period of"]):
        risk_level = "CRITICAL"
        risk_score = 95
        statutory_ref = "Section 27, Indian Contract Act 1872 (Restraint of Trade - VOID)"
        risk_reasons.append("Post-contractual restriction on trade or employment is void under Section 27 ICA 1872.")
        highlights.extend(["shall not engage", "non-compete", "prohibited from working"])

    # --- RULE 2: Unilateral Restraint of Legal Proceedings / Unfair Jurisdiction (Section 28 ICA 1872) ---
    elif any(k in text_lower for k in ["sole discretion of company", "waive right to court", "exclusive jurisdiction in foreign", "cannot sue", "binding unilateral arbitration"]):
        risk_level = "HIGH" if risk_level != "CRITICAL" else risk_level
        risk_score = max(risk_score, 85)
        statutory_ref = "Section 28, Indian Contract Act 1872 (Restraint of Legal Proceedings)"
        risk_reasons.append("Restricts consumer from legal recourse or enforces unfair unilateral jurisdiction.")
        highlights.extend(["waive right to court", "cannot sue", "sole discretion"])

    # --- RULE 3: Auto-Renewal Traps & Short Notice Windows ---
    if any(k in text_lower for k in ["automatically renew", "auto-renew", "auto renew", "recurring charge", "unless cancelled"]):
        if entities.get("deadline") != "N/A" or "written notice" in text_lower or "24 hours" in text_lower or "48 hours" in text_lower:
            risk_level = "HIGH" if risk_level not in ["CRITICAL"] else risk_level
            risk_score = max(risk_score, 80)
            risk_reasons.append("Auto-renewal clause with strict notice window or silent billing continuation.")
            highlights.extend(["automatically renew", "auto-renew", "unless cancelled"])
        else:
            risk_level = "MEDIUM" if risk_level == "LOW" else risk_level
            risk_score = max(risk_score, 55)
            risk_reasons.append("Automatic renewal active; requires affirmative cancellation.")

    # --- RULE 4: Liquidated Damages / Heavy Cancellation Penalty (Section 74 ICA 1872) ---
    if entities.get("amount") != "N/A" and any(k in text_lower for k in ["fee", "penalty", "charge", "liquidated damages", "forfeit"]):
        risk_score = max(risk_score, 75)
        if risk_level == "LOW":
            risk_level = "MEDIUM"
        if any(k in text_lower for k in ["penalty", "forfeit", "liquidated damages", "non-refundable"]):
            risk_level = "HIGH" if risk_level != "CRITICAL" else risk_level
            risk_score = max(risk_score, 85)
            statutory_ref = "Section 74, Indian Contract Act 1872 (Liquidated Damages vs Unreasonable Penalty)"
            risk_reasons.append(f"Financial penalty extracted ({entities.get('currency')} {entities.get('amount')}). Section 74 limits damages to reasonable compensation.")
            highlights.extend([entities.get("amount"), "penalty", "liquidated damages"])

    # --- RULE 5: No Refund / Strict Forfeiture Traps ---
    if any(k in text_lower for k in ["no refund", "non-refundable", "no cash refund", "forfeiture of deposit"]):
        risk_level = "HIGH" if risk_level != "CRITICAL" else risk_level
        risk_score = max(risk_score, 80)
        risk_reasons.append("Non-refundable term prevents consumer recovery even upon service default or early termination.")
        highlights.extend(["no refund", "non-refundable", "forfeiture"])

    # --- RULE 6: Unilateral Terms / Price Modification ---
    if any(k in text_lower for k in ["modify at any time", "change prices without notice", "sole discretion to alter", "reserve the right to amend"]):
        risk_level = "HIGH" if risk_level != "CRITICAL" else risk_level
        risk_score = max(risk_score, 75)
        risk_reasons.append("Unilateral right to alter price or contract terms without explicit consumer consent.")
        highlights.extend(["modify at any time", "without notice", "sole discretion"])

    # --- RULE 7: Cap on Liability / Total Exemption ---
    if any(k in text_lower for k in ["limitation of liability", "shall not exceed", "cap on liability", "not liable for any damage"]):
        if risk_level == "LOW":
            risk_level = "MEDIUM"
        risk_score = max(risk_score, 60)
        risk_reasons.append("Limits vendor liability for losses or damages incurred by consumer.")
        highlights.extend(["limitation of liability", "shall not exceed", "not liable"])

    if not risk_reasons:
        risk_reasons.append("Standard legal term; no unconscionable consumer risk detected.")

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "statutory_reference": statutory_ref,
        "risk_reasons": risk_reasons,
        "highlight_phrases": list(set(highlights))
    }


def analyze_contract_text(full_text: str) -> Dict[str, Any]:
    """
    Main entry point to perform complete contract NLP analysis.
    """
    clauses = segment_clauses(full_text)
    if not clauses:
        return {
            "error": "No valid clause text could be extracted from input."
        }

    analyzed_clauses = []
    critical_count = 0
    high_count = 0
    medium_count = 0
    low_count = 0

    financial_items = []
    top_alerts = []

    total_risk_sum = 0

    for item in clauses:
        c_text = item["text"]
        
        # 1. Classification
        cls_res = classify_clause(c_text)
        
        # 2. Entity Extraction
        entities = extract_financial_terms(c_text, category=cls_res["unified_category"])
        
        # 3. Risk & Indian Legal Evaluation
        risk_res = evaluate_risk_and_indian_law(c_text, cls_res["unified_category"], entities)

        r_level = risk_res["risk_level"]
        r_score = risk_res["risk_score"]

        total_risk_sum += r_score

        if r_level == "CRITICAL":
            critical_count += 1
            top_alerts.append(f"[{item['title']}] CRITICAL: {risk_res['risk_reasons'][0]}")
        elif r_level == "HIGH":
            high_count += 1
            top_alerts.append(f"[{item['title']}] HIGH RISK: {risk_res['risk_reasons'][0]}")
        elif r_level == "MEDIUM":
            medium_count += 1
        else:
            low_count += 1

        if entities.get("amount") != "N/A" or entities.get("refund_condition") != "N/A" or entities.get("consequence") != "N/A":
            financial_items.append({
                "clause_id": item["clause_id"],
                "clause_title": item["title"],
                "amount": entities.get("amount"),
                "currency": entities.get("currency"),
                "trigger": entities.get("trigger"),
                "deadline": entities.get("deadline"),
                "consequence": entities.get("consequence"),
                "refund_condition": entities.get("refund_condition"),
                "affected_party": entities.get("affected_party")
            })

        analyzed_clauses.append({
            "clause_id": item["clause_id"],
            "title": item["title"],
            "text": c_text,
            "classification": cls_res,
            "entities": entities,
            "risk": risk_res
        })

    num_clauses = len(analyzed_clauses)
    avg_risk = total_risk_sum / num_clauses if num_clauses > 0 else 0

    # Contract Health Score (100 = safest, 0 = most hazardous)
    contract_health_score = max(0, min(100, int(100 - avg_risk)))

    overall_verdict = "SAFE / TRANSPARENT"
    if contract_health_score < 40:
        overall_verdict = "HAZARDOUS / HIGH CONSUMER RISK"
    elif contract_health_score < 70:
        overall_verdict = "CAUTION ADVISED / MODERATE RISK"
    elif contract_health_score < 85:
        overall_verdict = "FAIR WITH MINOR CLAUSES"

    return {
        "contract_health_score": contract_health_score,
        "overall_verdict": overall_verdict,
        "stats": {
            "total_clauses": num_clauses,
            "critical_risk_count": critical_count,
            "high_risk_count": high_count,
            "medium_risk_count": medium_count,
            "low_risk_count": low_count,
            "financial_entities_found": len(financial_items)
        },
        "top_alerts": top_alerts[:5],
        "financial_obligations": financial_items,
        "clauses": analyzed_clauses
    }


def semantic_search_clauses(query: str, target_clauses: List[Dict[str, Any]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Dense Semantic Search over provided document clauses OR indexed ACORD corpus.
    """
    cache = get_models()
    sbert = cache.get("sbert")

    if not sbert:
        return []

    # Query embedding
    q_emb = sbert.encode(query, convert_to_tensor=True, normalize_embeddings=True)

    results = []

    if target_clauses and len(target_clauses) > 0:
        texts = [c["text"] for c in target_clauses]
        corpus_emb = sbert.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        scores = util.cos_sim(q_emb, corpus_emb)[0]

        top_indices = torch.topk(scores, k=min(top_k, len(texts)))
        for score, idx in zip(top_indices.values, top_indices.indices):
            i = idx.item()
            c = target_clauses[i]
            results.append({
                "score": round(float(score.item()), 4),
                "source": "Document Clause",
                "clause_id": c["clause_id"],
                "title": c["title"],
                "text": c["text"]
            })
    elif cache.get("acord_embeddings") is not None and cache.get("acord_metadata") is not None:
        acord_emb = cache["acord_embeddings"].to(q_emb.device)
        meta = cache["acord_metadata"]
        scores = util.cos_sim(q_emb, acord_emb)[0]

        top_indices = torch.topk(scores, k=min(top_k, len(meta["clause_ids"])))
        for score, idx in zip(top_indices.values, top_indices.indices):
            i = idx.item()
            results.append({
                "score": round(float(score.item()), 4),
                "source": "ACORD Corpus Index",
                "clause_id": meta["clause_ids"][i],
                "title": f"ACORD Corpus Clause {meta['clause_ids'][i]}",
                "text": meta["clause_texts"][i]
            })

    return results


if __name__ == "__main__":
    sample_text = """
    1. AUTOMATIC RENEWAL AND BILLING.
    Unless cancelled at least 48 hours prior to the expiration of the current monthly billing term, 
    this Agreement shall automatically renew for an additional 12-month period. A recurring subscription 
    fee of ₹4,999 will be deducted automatically from your linked credit card.

    2. CANCELLATION PENALTY & LIQUIDATED DAMAGES.
    If the customer terminates this service prior to the completion of the term, the user shall forfeit 
    their deposit and incur a cancellation fee of Rs 3,000 as liquidated damages within 15 days. No cash refunds are provided.

    3. RESTRAINT OF TRADE.
    The subscriber agrees that during the term and for 2 years post termination, the subscriber shall not engage in any competing legal or software business within India.

    4. LIMITATION OF LIABILITY.
    In no event shall the Company's cumulative liability for any loss or damage exceed $50 USD.
    """
    print("--- RUNNING SAMPLE CONTRACT ANALYSIS ---")
    output = analyze_contract_text(sample_text)
    print(json.dumps(output, indent=2))
