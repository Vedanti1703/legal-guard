"""
tests/test_risk_filter.py

Pytest suite for Two-Stage Risk Gating Filter:
- Keyword gate behavior
- Model-gate confidence threshold behavior
- Known safe clause is NOT flagged
- Known risky clause IS flagged
- Statutory grounding (ICA 1872 Sections 27, 28, 74)
"""

import pytest
from src.inference.risk_filter import (
    evaluate_two_stage_risk,
    match_keywords,
    filter_clauses_for_output,
    load_risk_keywords,
    load_risk_filter_config
)


def test_risk_keywords_loaded():
    kw = load_risk_keywords()
    assert "auto_renewal" in kw
    assert "penalty_fees" in kw
    assert "restraint" in kw
    assert "legal_recourse" in kw
    assert len(kw["auto_renewal"]["patterns"]) > 0


def test_known_safe_clause_not_flagged():
    """Known safe clauses must NOT be flagged by the gated pipeline."""
    safe_clause = (
        "All applicable taxes, including Goods and Services Tax (GST) at 18%, "
        "are itemized and displayed on your monthly tax invoice."
    )
    # Model predicts other_clause with high confidence
    res = evaluate_two_stage_risk(
        clause_text=safe_clause,
        predicted_category="other_clause",
        confidence=0.92,
        entities={"amount": "N/A", "deadline": "N/A"}
    )
    assert res["is_flagged"] is False
    assert res["risk_level"] == "LOW"
    assert res["risk_score"] < 40


def test_known_safe_privacy_term_not_flagged():
    safe_clause = (
        "We process user personal data strictly in accordance with our Privacy Policy "
        "and the Digital Personal Data Protection Act 2023."
    )
    res = evaluate_two_stage_risk(
        clause_text=safe_clause,
        predicted_category="other_clause",
        confidence=0.88
    )
    assert res["is_flagged"] is False
    assert res["risk_level"] == "LOW"


def test_known_risky_autorenewal_flagged():
    """Known risky auto-renewal clause with recurring billing must be flagged."""
    risky_clause = (
        "Unless cancelled by written notice at least 60 days prior to term expiration, "
        "this Agreement shall automatically renew for additional 12-month periods with recurring charge of ₹19,999."
    )
    res = evaluate_two_stage_risk(
        clause_text=risky_clause,
        predicted_category="Renewal Term",
        confidence=0.85,
        entities={"amount": "19,999", "currency": "₹", "deadline": "60 days"}
    )
    assert res["is_flagged"] is True
    assert res["risk_level"] in ["HIGH", "CRITICAL"]
    assert res["risk_score"] >= 70
    assert any("automatically renew" in p.lower() or "auto-renew" in p.lower() or "recurring charge" in p.lower() for p in res["highlight_phrases"])


def test_statutory_noncompete_section_27():
    """Restraint of trade is void under Section 27 and must trigger CRITICAL risk."""
    clause = "The employee shall not engage in any competing software or consulting business in India for 2 years."
    res = evaluate_two_stage_risk(
        clause_text=clause,
        predicted_category="other_clause",
        confidence=0.45
    )
    assert res["is_flagged"] is True
    assert res["risk_level"] == "CRITICAL"
    assert res["risk_score"] >= 90
    assert "Section 27" in res["statutory_reference"]


def test_statutory_court_waiver_section_28():
    """Restraint of legal proceedings is void under Section 28 and must trigger HIGH/CRITICAL risk."""
    clause = "The customer hereby waives right to court and consumer forum proceedings, agreeing to binding unilateral arbitration."
    res = evaluate_two_stage_risk(
        clause_text=clause,
        predicted_category="other_clause",
        confidence=0.50
    )
    assert res["is_flagged"] is True
    assert res["risk_level"] in ["HIGH", "CRITICAL"]
    assert "Section 28" in res["statutory_reference"]


def test_statutory_liquidated_damages_section_74():
    """Liquidated damages penalty must be grounded under Section 74."""
    clause = "Upon early cancellation, customer shall pay a penalty of Rs 10,000 as liquidated damages within 7 days."
    res = evaluate_two_stage_risk(
        clause_text=clause,
        predicted_category="Liquidated Damages",
        confidence=0.78,
        entities={"amount": "10,000", "currency": "Rs", "deadline": "7 days"}
    )
    assert res["is_flagged"] is True
    assert res["risk_level"] in ["HIGH", "CRITICAL"]
    assert "Section 74" in res["statutory_reference"]


def test_model_gate_threshold_behavior():
    """A risky category clause with confidence below threshold and no statutory trigger should not pass."""
    text = "The contract term may be extended upon mutual discussions."
    config = {"model_confidence_threshold": 0.60, "keyword_gate_enabled": True}
    
    # Below threshold (0.35 < 0.60)
    res_low = evaluate_two_stage_risk(
        clause_text=text,
        predicted_category="Renewal Term",
        confidence=0.35,
        config=config
    )
    assert res_low["gating_details"]["passes_model_gate"] is False


def test_filter_clauses_for_output():
    clauses = [
        {"id": 1, "risk": {"risk_level": "LOW"}},
        {"id": 2, "risk": {"risk_level": "MEDIUM"}},
        {"id": 3, "risk": {"risk_level": "HIGH"}},
        {"id": 4, "risk": {"risk_level": "CRITICAL"}},
        {"id": 5, "risk": {"risk_level": "LOW"}},
    ]
    # Default: min_risk_level = MEDIUM -> 3 visible, 2 hidden
    vis, hidden_count = filter_clauses_for_output(clauses, min_risk_level="MEDIUM", include_safe=False)
    assert len(vis) == 3
    assert hidden_count == 2

    # Include safe = True -> all 5 visible, 0 hidden
    vis_all, hidden_all = filter_clauses_for_output(clauses, min_risk_level="MEDIUM", include_safe=True)
    assert len(vis_all) == 5
    assert hidden_all == 0
