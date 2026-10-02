"""
tests/test_summarizer.py

Pytest suite for Whole-Document Summarizer:
- Output schema verification (overview, key_points, word counts, compression ratio)
- Extractive, hybrid, abstractive modes
- Edge cases: very short docs, empty docs error handling, non-English warning
"""

import pytest
from src.summarization.summarizer import summarize_document, split_into_sentences


SAMPLE_CONTRACT_TEXT = """
1. SUBSCRIPTION TERM & AUTOMATIC RENEWAL.
The Initial Subscription Term shall commence on the Effective Date and continue for twelve (12) consecutive months. 
Unless cancelled by written notice at least sixty (60) days prior to the expiration of the current term, 
this Agreement shall automatically renew for additional 12-month periods. 
A recurring annual subscription charge of ₹24,999 will be automatically debited from the Customer's registered payment method.

2. CANCELLATION PENALTY & FORFEITURE.
If the Customer terminates this Agreement prior to the expiration of the subscription period, 
the Customer shall incur a cancellation fee of Rs 10,000 as liquidated damages within 7 business days. 
All accrued fees and advance payments shall be non-refundable.

3. RESTRAINT OF PROFESSION & NON-COMPETE.
During the subscription term and for a period of two (2) years following any termination hereof, 
the Customer and its affiliates shall not engage in any software product or business that competes with the Company's services in India.

4. GOVERNING LAW & DISPUTES.
This contract is governed by the laws of India. All disputes shall be subject to the exclusive jurisdiction of courts in New Delhi.
"""


def test_summarizer_output_schema_extractive():
    res = summarize_document(SAMPLE_CONTRACT_TEXT, mode="extractive", length="medium")
    assert "error" not in res
    assert "overview" in res
    assert isinstance(res["overview"], str)
    assert len(res["overview"]) > 20

    assert "key_points" in res
    assert isinstance(res["key_points"], dict)
    assert "purpose" in res["key_points"] or "renewal" in res["key_points"]

    assert "word_count_original" in res
    assert "word_count_summary" in res
    assert "compression_ratio" in res
    assert 0.0 < res["compression_ratio"] <= 1.0
    assert res["word_count_summary"] < res["word_count_original"]


def test_summarizer_output_schema_hybrid():
    res = summarize_document(SAMPLE_CONTRACT_TEXT, mode="hybrid", length="short")
    assert "error" not in res
    assert "overview" in res
    assert "key_points" in res
    assert res["length"] == "short"


def test_summarizer_financial_dates_extraction():
    res = summarize_document(SAMPLE_CONTRACT_TEXT, mode="extractive", length="medium")
    assert "important_dates_and_amounts" in res
    dates_and_amts = res["important_dates_and_amounts"]
    assert len(dates_and_amts) > 0
    # Check that 24,999 or 10,000 was extracted
    amounts = [str(d.get("amount", "")).replace(",", "") for d in dates_and_amts if d.get("amount") != "N/A"]
    assert any("24999" in a or "10000" in a for a in amounts)


def test_edge_case_empty_document():
    res = summarize_document("", mode="extractive")
    assert "error" in res
    assert "unreadable" in res["error"].lower() or "insufficient" in res["error"].lower()


def test_edge_case_short_document():
    short_text = "This is a brief non-disclosure clause regarding proprietary company trade secrets."
    res = summarize_document(short_text, mode="extractive")
    assert "error" not in res
    assert res["word_count_original"] < 60
    assert any("short" in w.lower() for w in res.get("warnings", []))


def test_edge_case_non_english():
    # Non-ASCII text sample
    non_english_text = "यह एक कानूनी अनुबंध है जो सेवा शर्तों और उपभोक्ता अधिकारों को परिभाषित करता है। " * 10
    res = summarize_document(non_english_text, mode="extractive")
    assert "error" not in res
    assert any("non-english" in w.lower() for w in res.get("warnings", []))
