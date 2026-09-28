"""
src/preprocessing/extract_financial_entities.py

Rule-based Information Extraction pipeline for consumer contract terms.
Extracts structured financial and unfavorable condition entities without hallucinated NER:
- amount & currency (INR, Rs, ₹, USD, $, %, etc.)
- trigger condition
- deadline / notice window
- duration
- action required
- consequence / penalty
- refund condition
- affected party
"""

import re
import sys
import json
from typing import Dict, Any

# Ensure UTF-8 console output for Indian Rupee symbol and special characters on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


CURRENCY_PATTERNS = [
    (r'(?:₹|INR|Rs\.?|Rupees)\s*([\d,]+(?:\.\d{1,2})?)', "INR"),
    (r'(?:\$|USD)\s*([\d,]+(?:\.\d{1,2})?)', "USD"),
    (r'(?:€|EUR)\s*([\d,]+(?:\.\d{1,2})?)', "EUR"),
    (r'([\d,]+(?:\.\d{1,2})?)\s*(?:%|percent|per\s*cent)', "PERCENT")
]

DEADLINE_PATTERNS = [
    r'(?:within|prior to|at least|before|no later than)\s+(\d+\s*(?:hours?|days?|weeks?|months?|business days?))',
    r'(\d+\s*(?:hours?|days?|weeks?|months?))\s+(?:prior|before|advance notice|written notice)'
]

TRIGGER_PATTERNS = [
    r'(cancellation\s+(?:made|requested|received)?\s*(?:within|prior|after)?[^\.,;:]*)',
    r'(failure to\s+(?:pay|notify|renew|comply)[^\.,;:]*)',
    r'(in the event of\s+[^\.,;:]*)',
    r'(upon\s+(?:termination|default|expiration|breach)[^\.,;:]*)',
    r'(if the (?:customer|subscriber|user|buyer)\s+[^\.,;:]*)'
]

CONSEQUENCE_PATTERNS = [
    r'(incur\s+(?:a\s+)?(?:fee|charge|penalty)\s+of[^\.,;:]*)',
    r'((?:forfeit|forfeiture of)\s+[^\.,;:]*)',
    r'((?:charged|levied|deducted)\s+an?\s+(?:additional\s+)?(?:fee|penalty|charge|amount)[^\.,;:]*)',
    r'(liquidated damages\s+of[^\.,;:]*)',
    r'(automatically\s+renew(?:ed|s)?[^\.,;:]*)',
    r'(no\s+refund[^\.,;:]*)'
]

REFUND_PATTERNS = [
    r'(no\s+(?:cash\s+)?refunds?(?:\s+will\s+be\s+provided)?)',
    r'(non-refundable)',
    r'(refund(?:ed)?\s+only\s+(?:as|in\s+the\s+form\s+of)\s+[^\.,;:]*)',
    r'(subject to\s+(?:a\s+)?cancellation\s+charge[^\.,;:]*)'
]


def extract_financial_terms(clause_text: str, category: str = "unspecified") -> Dict[str, Any]:
    """
    Parses a legal clause text and extracts financial consequence entities.
    """
    if not clause_text or not isinstance(clause_text, str):
        return {}

    amount = "N/A"
    currency = "N/A"
    for pattern, curr in CURRENCY_PATTERNS:
        match = re.search(pattern, clause_text, re.IGNORECASE)
        if match:
            amount = match.group(1).replace(",", "").strip()
            currency = curr
            break

    deadline = "N/A"
    for dp in DEADLINE_PATTERNS:
        match = re.search(dp, clause_text, re.IGNORECASE)
        if match:
            deadline = match.group(1).strip()
            break

    trigger = "N/A"
    for tp in TRIGGER_PATTERNS:
        match = re.search(tp, clause_text, re.IGNORECASE)
        if match:
            trigger = match.group(1).strip()
            break

    consequence = "N/A"
    for cp in CONSEQUENCE_PATTERNS:
        match = re.search(cp, clause_text, re.IGNORECASE)
        if match:
            consequence = match.group(1).strip()
            break

    refund_cond = "N/A"
    for rp in REFUND_PATTERNS:
        match = re.search(rp, clause_text, re.IGNORECASE)
        if match:
            refund_cond = match.group(1).strip()
            break

    # Affected party detection
    party = "consumer"
    if re.search(r'\b(employee)\b', clause_text, re.IGNORECASE):
        party = "employee"
    elif re.search(r'\b(distributor|reseller|licensee)\b', clause_text, re.IGNORECASE):
        party = "commercial_counterparty"
    elif re.search(r'\b(customer|subscriber|user|passenger|buyer|consumer)\b', clause_text, re.IGNORECASE):
        party = "consumer"

    return {
        "amount": amount,
        "currency": currency,
        "trigger": trigger,
        "deadline": deadline,
        "consequence": consequence,
        "refund_condition": refund_cond,
        "affected_party": party,
        "category": category,
        "raw_text": clause_text
    }


def demo_financial_extraction():
    examples = [
        ("Cancellation made within 24 hours of departure will incur a fee of ₹3,000.", "cancellation_fee"),
        ("Unless cancelled at least 48 hours prior to the billing cycle, the subscription will automatically renew and the monthly fee of Rs 299 will be charged.", "automatic_renewal"),
        ("Return requests must be initiated within 7 days of delivery. No cash refunds are provided; refund only as store credit.", "refund_restriction"),
        ("A late payment penalty of 2% per month will be levied on overdue invoices exceeding 15 days.", "penalty"),
        ("Liquidated damages in the sum of $500 per day of delay shall be payable by the contractor.", "liquidated_damages")
    ]

    print("--- DEMONSTRATION OF FINANCIAL INFORMATION EXTRACTION ---")
    for text, cat in examples:
        extracted = extract_financial_terms(text, category=cat)
        print(f"\nClause: \"{text}\"")
        print("Structured Output:")
        print(json.dumps(extracted, indent=2))


if __name__ == "__main__":
    demo_financial_extraction()
