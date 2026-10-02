"""
tests/test_api_smoke.py

Smoke test suite for web server API endpoints:
- GET /api/health
- GET /api/models
- GET /api/samples
- POST /api/analyze (JSON text payload, include_safe toggle, min_risk_level)
- POST /api/summarize (modes: hybrid, extractive, length options)
- POST /api/search (dense semantic retrieval)
"""

import pytest
import json
from server import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data


def test_api_models(client):
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.get_json()
    assert "active_model" in data
    assert "models" in data
    assert "legalbert" in data["models"]
    assert "tfidf" in data["models"]


def test_api_samples(client):
    res = client.get("/api/samples")
    assert res.status_code == 200
    data = res.get_json()
    assert "samples" in data
    assert "saas_autorenewal" in data["samples"]


def test_api_analyze_smoke(client):
    sample_text = """
    1. AUTOMATIC RENEWAL.
    Unless cancelled 30 days prior to expiry, this subscription automatically renews for 12 months with recurring fee of ₹12,000.
    
    2. CANCELLATION PENALTY.
    Early cancellation incurs a penalty fee of Rs 3,000 as liquidated damages. No cash refunds.
    
    3. GOVERNING LAW.
    This agreement is governed by the laws of India.
    """
    res = client.post("/api/analyze", json={"text": sample_text, "include_safe": False})
    assert res.status_code == 200
    data = res.get_json()
    assert "contract_health_score" in data
    assert "overall_verdict" in data
    assert "stats" in data
    assert "clauses" in data
    # Safe clause (Governing law) should be hidden by default
    assert data.get("safe_clauses_hidden", 0) >= 1


def test_api_analyze_include_safe_toggle(client):
    sample_text = """
    1. AUTOMATIC RENEWAL.
    Unless cancelled 30 days prior to expiry, this subscription automatically renews.
    
    2. GOVERNING LAW.
    This agreement is governed by the laws of India.
    """
    res_hidden = client.post("/api/analyze?include_safe=false", json={"text": sample_text})
    data_hidden = res_hidden.get_json()

    res_all = client.post("/api/analyze?include_safe=true", json={"text": sample_text})
    data_all = res_all.get_json()

    # When include_safe=true, more or equal clauses must be returned
    assert len(data_all["clauses"]) >= len(data_hidden["clauses"])


def test_api_summarize_smoke(client):
    sample_text = """
    This SaaS Services Agreement is entered into by and between Alpha Corp and Beta Ltd.
    Alpha Corp agrees to provide cloud monitoring services for an initial term of 12 months.
    The customer shall pay a monthly subscription fee of $500 USD within 15 days of invoice.
    Either party may terminate upon 30 days written notice.
    Alpha Corp liability shall not exceed the fees paid in the preceding 6 months.
    Governing law shall be the laws of India.
    """
    res = client.post("/api/summarize?mode=extractive&length=medium", json={"text": sample_text})
    assert res.status_code == 200
    data = res.get_json()
    assert "overview" in data
    assert "key_points" in data
    assert "word_count_original" in data
    assert "compression_ratio" in data


def test_api_search_smoke(client):
    res = client.post("/api/search", json={
        "query": "cancellation fee or penalty",
        "clauses": [
            {"title": "Clause 1", "text": "Penalty fee of Rs 5000 is required upon early cancellation."},
            {"title": "Clause 2", "text": "Customer service is open 9 to 5 daily."}
        ]
    })
    assert res.status_code == 200
    data = res.get_json()
    assert "results" in data
    assert len(data["results"]) > 0
    # Top result should be the penalty clause
    assert "Penalty" in data["results"][0]["text"]
