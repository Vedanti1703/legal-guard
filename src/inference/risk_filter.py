"""
src/inference/risk_filter.py

Two-stage gating pipeline and consumer risk scoring module:
1. Stage 1: Model Gate (category in RISKY_CATEGORIES and confidence >= MODEL_CONF_THRESHOLD)
2. Stage 2: Keyword Gate (regex pattern matching from data/custom/risk_keywords.json)
3. Stage 3: Severity Scoring (0-100 score + Indian Contract Act grounding Sec 27, 28, 74)
4. Stage 4: Output Filtering (suppress safe clauses by default, return count of hidden safe clauses)
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger("risk_filter")

# Categories deemed risky (everything except non-risky background clauses)
RISKY_CATEGORIES = {
    "Liquidated Damages",
    "Notice Period To Terminate Renewal",
    "Post-Termination Services",
    "Renewal Term",
    "Revenue/Profit Sharing",
    "Termination For Convenience",
    "penalty",
    "renewal_notice",
    "automatic_renewal",
    "revenue_sharing",
    "termination",
    "liability_limitation",
    "price_increase_condition",
    "other_consumer_risk"
}

SAFE_CATEGORIES = {"other_clause", "unmapped", "none", "standard_term"}

DEFAULT_CONFIG = {
    "model_confidence_threshold": 0.40,
    "keyword_gate_enabled": True,
    "min_risk_level": "MEDIUM",
    "min_recall_floor": 0.70
}

_KEYWORD_PATTERNS_CACHE: Optional[Dict[str, Any]] = None
_FILTER_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def load_risk_keywords() -> Dict[str, Any]:
    """Loads and compiles regular expressions from risk_keywords.json."""
    global _KEYWORD_PATTERNS_CACHE
    if _KEYWORD_PATTERNS_CACHE is not None:
        return _KEYWORD_PATTERNS_CACHE

    kw_file = PROJECT_ROOT / "data" / "custom" / "risk_keywords.json"
    if not kw_file.exists():
        logger.warning(f"Risk keywords file not found at {kw_file}. Using built-in fallbacks.")
        raw_data = {"categories": {}}
    else:
        with open(kw_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

    compiled = {}
    for cat_key, cat_val in raw_data.get("categories", {}).items():
        patterns = []
        for pat_str in cat_val.get("patterns", []):
            try:
                patterns.append(re.compile(pat_str, re.IGNORECASE))
            except re.error as e:
                logger.error(f"Invalid regex pattern '{pat_str}' in {cat_key}: {e}")
        compiled[cat_key] = {
            "name": cat_val.get("name", cat_key),
            "severity_weight": cat_val.get("severity_weight", 0.7),
            "default_level": cat_val.get("default_level", "MEDIUM"),
            "patterns": patterns,
            "reason": cat_val.get("reason", "Potential risk detected."),
            "suggested_action": cat_val.get("suggested_action", "Review clause carefully.")
        }

    _KEYWORD_PATTERNS_CACHE = compiled
    return compiled


def load_risk_filter_config() -> Dict[str, Any]:
    """Loads tuned threshold parameters from outputs/models/risk_filter_config.json."""
    global _FILTER_CONFIG_CACHE
    if _FILTER_CONFIG_CACHE is not None:
        return _FILTER_CONFIG_CACHE

    cfg_file = PROJECT_ROOT / "outputs" / "models" / "risk_filter_config.json"
    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                _FILTER_CONFIG_CACHE = {**DEFAULT_CONFIG, **cfg}
                return _FILTER_CONFIG_CACHE
        except Exception as e:
            logger.warning(f"Failed to load risk filter config from {cfg_file}: {e}")

    _FILTER_CONFIG_CACHE = DEFAULT_CONFIG.copy()
    return _FILTER_CONFIG_CACHE


def match_keywords(clause_text: str) -> List[Dict[str, Any]]:
    """
    Evaluates clause text against keyword regex patterns with word boundaries.
    Returns matched categories, matched text snippets, and weights.
    """
    categories_cfg = load_risk_keywords()
    matches = []

    for cat_key, cat_data in categories_cfg.items():
        matched_spans = []
        for pat in cat_data["patterns"]:
            for m in pat.finditer(clause_text):
                matched_spans.append(m.group(0))

        if matched_spans:
            matches.append({
                "category_key": cat_key,
                "name": cat_data["name"],
                "weight": cat_data["severity_weight"],
                "default_level": cat_data["default_level"],
                "reason": cat_data["reason"],
                "suggested_action": cat_data["suggested_action"],
                "matched_tokens": list(set(matched_spans))
            })

    return matches


def evaluate_two_stage_risk(
    clause_text: str,
    predicted_category: str,
    confidence: float,
    entities: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Executes the two-stage gating pipeline and severity calculation:
    Stage 1: Model Gate
    Stage 2: Keyword Gate
    Stage 3: Severity Scoring + Statutory Grounding
    """
    if config is None:
        config = load_risk_filter_config()

    if entities is None:
        entities = {}

    model_threshold = config.get("model_confidence_threshold", 0.40)
    keyword_gate_active = config.get("keyword_gate_enabled", True)

    text_lower = clause_text.lower()
    keyword_matches = match_keywords(clause_text)

    # 1. Stage 1: Model Gate Evaluation
    is_model_risky_cat = (predicted_category in RISKY_CATEGORIES) and (predicted_category not in SAFE_CATEGORIES)
    passes_model_gate = is_model_risky_cat and (confidence >= model_threshold)

    # 2. Stage 2: Keyword Gate Evaluation
    passes_keyword_gate = len(keyword_matches) > 0 if keyword_gate_active else True

    # High-severity override (e.g. non-compete Sec 27 or illegal jurisdiction Sec 28)
    has_statutory_critical = any(
        m["category_key"] == "restraint" or m["category_key"] == "legal_recourse"
        for m in keyword_matches
    )

    # Clause is flagged only if both signals agree, OR if a strict statutory critical pattern matches
    is_flagged = (passes_model_gate and passes_keyword_gate) or has_statutory_critical

    # 3. Stage 3: Severity Scoring (0 - 100)
    # Base score
    risk_score = 10.0
    risk_level = "LOW"
    statutory_ref = "N/A"
    risk_reasons = []
    suggested_actions = []
    highlights = []

    for km in keyword_matches:
        highlights.extend(km["matched_tokens"])
        risk_reasons.append(km["reason"])
        suggested_actions.append(km["suggested_action"])

    # Statutory Rule 1: Section 27 ICA 1872 (Restraint of Trade / Profession)
    if any(k in text_lower for k in ["shall not engage", "restraint of trade", "non-compete", "prohibited from working", "competing business"]):
        risk_score = 95.0
        risk_level = "CRITICAL"
        statutory_ref = "Section 27, Indian Contract Act 1872 (Restraint of Trade - VOID)"
        risk_reasons.insert(0, "Post-contractual restriction on trade or employment is void under Section 27 ICA 1872.")
        suggested_actions.insert(0, "Remove non-compete restriction; unenforceable under settled Indian Supreme Court jurisprudence.")

    # Statutory Rule 2: Section 28 ICA 1872 (Restraint of Legal Proceedings)
    elif any(k in text_lower for k in ["sole discretion of company", "waive right to court", "cannot sue", "binding unilateral arbitration", "waive jury trial", "exclusive jurisdiction in foreign"]):
        risk_score = max(risk_score, 85.0)
        risk_level = "HIGH" if risk_level != "CRITICAL" else risk_level
        statutory_ref = "Section 28, Indian Contract Act 1872 (Restraint of Legal Proceedings)"
        risk_reasons.insert(0, "Deprives consumer of jurisdiction in judicial courts or consumer disputes commissions.")
        suggested_actions.insert(0, "Dispute clause cannot oust consumer forum jurisdiction under Consumer Protection Act 2019.")

    # Statutory Rule 3: Section 74 ICA 1872 (Liquidated Damages vs Penalty)
    elif entities.get("amount") not in [None, "N/A"] and any(k in text_lower for k in ["penalty", "liquidated damages", "cancellation fee", "forfeit", "forfeiture"]):
        risk_score = max(risk_score, 82.0)
        risk_level = "HIGH" if risk_level not in ["CRITICAL"] else risk_level
        amt_str = f"{entities.get('currency', '₹')} {entities.get('amount')}"
        statutory_ref = "Section 74, Indian Contract Act 1872 (Compensation for Breach of Contract)"
        risk_reasons.insert(0, f"Exorbitant penalty or liquidated damages stipulated ({amt_str}). Sec 74 restricts damages to reasonable proved loss.")
        suggested_actions.insert(0, "Limit fee to actual direct administrative expense rather than punitive forfeiture.")

    # Score aggregation if not overridden by statutory rule
    if statutory_ref == "N/A":
        if is_flagged:
            # Aggregate weights
            max_kw_weight = max([km["weight"] for km in keyword_matches], default=0.5)
            # Combine model confidence (0-1) and keyword weight (0-1)
            combined_factor = (max_kw_weight * 0.6) + (confidence * 0.4)
            computed_score = 45.0 + (combined_factor * 45.0)

            # Boost if money or deadline entity is present
            if entities.get("amount") not in [None, "N/A"]:
                computed_score += 8.0
            if entities.get("deadline") not in [None, "N/A"]:
                computed_score += 5.0

            risk_score = min(92.0, computed_score)

            if risk_score >= 80:
                risk_level = "HIGH"
            elif risk_score >= 50:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
        else:
            # Gating suppressed this clause
            risk_score = 15.0 if len(keyword_matches) > 0 else 5.0
            risk_level = "LOW"
            if not risk_reasons:
                risk_reasons.append("Standard legal term; no unconscionable risk detected.")
            if not suggested_actions:
                suggested_actions.append("No action required.")

    # Deduplicate reasons, highlights, actions
    unique_reasons = list(dict.fromkeys(risk_reasons))
    unique_actions = list(dict.fromkeys(suggested_actions))
    unique_highlights = list(set(highlights))

    # Determine final gating status
    gated_passed = (risk_level in ["MEDIUM", "HIGH", "CRITICAL"]) and is_flagged

    return {
        "is_flagged": gated_passed,
        "risk_level": risk_level,
        "risk_score": round(float(risk_score), 1),
        "statutory_reference": statutory_ref,
        "risk_reasons": unique_reasons,
        "suggested_actions": unique_actions,
        "highlight_phrases": unique_highlights,
        "gating_details": {
            "passes_model_gate": passes_model_gate,
            "passes_keyword_gate": passes_keyword_gate,
            "has_statutory_critical": has_statutory_critical,
            "model_confidence": round(float(confidence), 4),
            "model_threshold_used": model_threshold,
            "keyword_matches_count": len(keyword_matches)
        }
    }


def filter_clauses_for_output(
    analyzed_clauses: List[Dict[str, Any]],
    min_risk_level: str = "MEDIUM",
    include_safe: bool = False
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Filters clauses so that only clauses with risk_level >= min_risk_level are returned by default.
    Returns: (visible_clauses, safe_clauses_hidden_count)
    """
    level_ranks = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    min_rank = level_ranks.get(min_risk_level.upper(), 2)

    visible = []
    hidden_count = 0

    for clause in analyzed_clauses:
        lvl = clause.get("risk", {}).get("risk_level", "LOW")
        rank = level_ranks.get(lvl, 1)

        if include_safe or rank >= min_rank:
            visible.append(clause)
        else:
            hidden_count += 1

    return visible, hidden_count
