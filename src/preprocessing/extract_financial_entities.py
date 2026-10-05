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
from typing import Dict, Any, List, Optional

# Ensure UTF-8 console output for Indian Rupee symbol and special characters on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


CURRENCY_PATTERNS = [
    # INR with symbol or words before number: ₹ 24,999 or Rs. 10,000 or INR 50,000
    (r'(?:₹|\bINR\b|\bRs\.?\b|\bRupees\b)\s*(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)', "INR"),
    # Amount followed by INR/Rupees: 24,999 Rupees or 10,000 INR
    (r'(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)\s*(?:\bINR\b|\bRs\.?\b|\bRupees\b)', "INR"),
    # USD: $ 100 or USD 100 or 100 USD
    (r'(?:\$|\bUSD\b)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)', "USD"),
    (r'(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)\s*(?:\$|\bUSD\b|\bDollars\b)', "USD"),
    # EUR: € 500 or EUR 500
    (r'(?:€|\bEUR\b)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)', "EUR"),
    (r'(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)\s*(?:€|\bEUR\b|\bEuros\b)', "EUR"),
    # Percentage: 18% or 18 percent or 20%
    (r'(\d+(?:\.\d{1,2})?)\s*(?:%|\bpercent\b|\bper\s*cent\b)', "PERCENT")
]

DEADLINE_PATTERNS = [
    # 1. "at least sixty (60) days prior to ...", "within 7 business days", "within 3 days of delivery", "within 72 hours of scheduled flight departure"
    r'((?:at\s+least|within|prior\s+to|before|no\s+later\s+than)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty|twenty-four|thirty|forty-five|sixty|seventy-two|ninety|180|365)\s*(?:\([^)]+\))?\s*(?:hours?|days?|weeks?|months?|years?|business\s+days?|calendar\s+days?)(?:\s+(?:prior(?:\s+to)?|before|after|from|of)\s+[^.;:\n]{2,35})?)',
    # 2. "upon 24 hours notice", "with 30 days written notice", "after 24 hours of delivery"
    r'((?:upon|with|after)\s+(?:\d+|one|two|three|four|five|six|seven|ten|fifteen|twenty|twenty-four|thirty|sixty)\s*(?:\([^)]+\))?\s*(?:hours?|days?|weeks?|months?)(?:\s+(?:written\s+)?notice|\s+of\s+[^.;:\n]{2,25})?)',
    # 3. "60 days prior notice", "30 days advance notice", "14 days notice"
    r'((?:\d+|one|two|three|four|five|six|seven|ten|fifteen|twenty|thirty|sixty|ninety)\s*(?:\([^)]+\))?\s*(?:hours?|days?|weeks?|months?|business\s+days?)\s+(?:prior|before|advance\s+notice|written\s+notice|notice\s+period))',
    # 4. "notice period of 30 days", "commitment of 24 months"
    r'(notice\s+period\s+of\s+\d+\s*(?:hours?|days?|weeks?|months?))',
    # 5. "exceeding 6 hours", "exceeding 15 days", "exceeding 30 days"
    r'(exceeding\s+\d+\s*(?:hours?|days?|weeks?|months?))',
    # 6. Generic "within X days"
    r'(within\s+\d+\s*(?:hours?|days?|weeks?|months?|business\s+days?|calendar\s+days?))'
]

TRIGGER_PATTERNS = [
    # 1. Unless cancelled by written notice
    r'(unless\s+cancelled\s+(?:by\s+written\s+notice)?[^.;\n]*)',
    r'(unless\s+terminated\s+[^.;\n]*)',
    # 2. Prior to expiration / completion
    r'(prior\s+to\s+(?:the\s+)?(?:expiration|completion)\s+of[^.;\n]*)',
    # 3. If customer/member/borrower/buyer terminates/defaults/fails
    r'(if\s+(?:the\s+)?(?:customer|subscriber|user|buyer|member|borrower|tenant|employee|party)\s+[^.;\n]*)',
    # 4. Failure to pay / notify / comply
    r'(failure\s+to\s+(?:pay|notify|renew|comply|repay|cure)[^.;\n]*)',
    # 5. Upon cancellation / early exit / termination / default / breach
    r'(upon\s+(?:cancellation|early\s+exit|termination|default|expiration|breach|delay|non-payment)[^.;\n]*)',
    # 6. In the event of flight delay / schedule modification / default / breach
    r'(in\s+the\s+event\s+of\s+[^.;\n]*)',
    # 7. Cancellations requested / made
    r'(cancellations?\s+(?:made|requested|received)?\s*(?:within|prior|after)?[^.;\n]*)',
    # 8. Claims / reports submitted
    r'(claims?\s+for\s+[^.;\n]*)',
    r'(reports?\s+submitted\s+[^.;\n]*)',
    # 9. In case of default / breach / late payment
    r'(in\s+case\s+of\s+(?:default|late\s+payment|breach|delay)[^.;\n]*)',
    # 10. Item returns
    r'(item\s+returns?\s+must\s+be\s+initiated[^.;\n]*)'
]

CONSEQUENCE_PATTERNS = [
    # 1. Incur fee / penalty / damages
    r'((?:shall|will|may)\s+(?:incur|be\s+liable\s+(?:for|to\s+pay)|pay)\s+[^.;\n]*(?:fee|penalty|charge|damages|fine|cost)[^.;\n]*)',
    r'(incur\s+(?:a\s+)?(?:cancellation|early\s+termination|exit|late)?\s*(?:fee|penalty|charge|damages)[^.;\n]*)',
    # 2. Levied / charged / deducted / debited
    r'((?:shall|will)\s+be\s+(?:levied|charged|deducted|withheld|debited|forfeited)[^.;\n]*)',
    r'((?:charged|levied|deducted|debited)\s+(?:an?\s+)?(?:additional\s+)?(?:fee|penalty|charge|amount|dues)[^.;\n]*)',
    # 3. Liquidated damages
    r'((?:as\s+)?(?:liquidated|liquidity)\s+damages[^.;\n]*)',
    # 4. Automatic renewal / debit
    r'((?:shall|will)\s+(?:automatically\s+renew|be\s+automatically\s+renewed|renew\s+automatically)[^.;\n]*)',
    r'(automatically\s+renew(?:ed|s|ing)?[^.;\n]*)',
    r'((?:will|shall)\s+be\s+automatically\s+debited[^.;\n]*)',
    # 5. Dues charged via auto-debit
    r'(dues\s+of\s+[^.;\n]*\s+will\s+be\s+charged[^.;\n]*)',
    # 6. Forfeiture / deposit loss / cancellation of allotment
    r'((?:shall\s+)?forfeit(?:ure\s+of)?\s+[^.;\n]*)',
    r'((?:loss\s+of|cancel(?:lation\s+of)?)\s+(?:deposit|allotment|booking|shares?)[^.;\n]*)',
    # 7. Interest / penal charge
    r'(attract\s+(?:a\s+)?(?:late\s+)?(?:interest|penalty|charge)[^.;\n]*)',
    r'((?:penal|late|compound)\s+interest\s+(?:of|rate|charge)[^.;\n]*)',
    # 8. Non-refundable / no refund
    r'((?:shall|will)\s+be\s+(?:strictly\s+)?non-refundable[^.;\n]*)',
    r'(strictly\s+non-refundable[^.;\n]*)',
    r'(non-refundable[^.;\n]*)',
    r'(no\s+(?:cash\s+)?refunds?[^.;\n]*)',
    # 9. Remedy / rebooking / restocking / rejected
    r'(sole\s+remedy\s+shall\s+be\s+[^.;\n]*)',
    r'(subject\s+to\s+(?:a\s+)?\d+%\s+restocking\s+fee[^.;\n]*)',
    r'(rejected\s+without\s+review[^.;\n]*)',
    # 10. Acceleration / loan recall
    r'((?:accelerate|acceleration\s+of)\s+(?:all\s+)?(?:debt|loan|dues|payments?)[^.;\n]*)'
]

REFUND_PATTERNS = [
    # 1. No cash refunds or pro-rated credits
    r'(no\s+(?:cash\s+)?refunds?(?:\s+or\s+pro-rated\s+credits)?(?:\s+(?:will|shall)\s+be\s+(?:issued|provided|processed|given))?[^.;\n]*)',
    # 2. No cash refund or hotel accommodation reimbursement
    r'(no\s+(?:cash\s+)?refund\s+or\s+[^.;\n]*reimbursement\s+shall\s+be\s+provided)',
    # 3. Strictly non-refundable
    r'((?:strictly\s+)?non-refundable(?:\s+(?:under\s+any\s+circumstances|deposit|advance|booking))?)',
    r'(all\s+accrued\s+fees[^.;\n]*shall\s+be\s+non-refundable)',
    # 4. Store credit / reward points / wallet only
    r'(refund(?:ed)?\s+(?:strictly\s+|only\s+)?(?:as|in\s+the\s+form\s+of)\s+[^.;\n]*)',
    r'(store\s+(?:reward\s+)?(?:credit|points|voucher)[^.;\n]*)',
    # 5. No bank refund
    r'(no\s+bank\s+refunds?(?:\s+or\s+cash\s+returns?)?[^.;\n]*)',
    # 6. Restocking fee / deduction
    r'(subject\s+to\s+(?:a\s+)?(?:cancellation|restocking|administrative)\s+(?:fee|charge|deduction)[^.;\n]*)',
    # 7. Forfeiture of deposit / advance
    r'((?:advance\s+payments?|earnest\s+money|security\s+deposit)\s+shall\s+be\s+forfeited[^.;\n]*)'
]


def clean_entity_text(val: str, max_words: int = 15) -> str:
    """Cleans up leading/trailing punctuation and trims length for compact display."""
    if not val or val == "N/A":
        return "N/A"
    cleaned = re.sub(r'\s+', ' ', val).strip(" .,;:-\"'\t\r\n")
    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words]) + "..."
    return cleaned


def extract_financial_terms(clause_text: str, category: str = "unspecified") -> Dict[str, Any]:
    """
    Parses a legal clause text and extracts financial consequence entities without hallucinations:
    - amount & currency
    - trigger condition
    - deadline / notice window
    - consequence / penalty
    - refund condition
    - affected party
    """
    if not clause_text or not isinstance(clause_text, str):
        return {}

    amount = "N/A"
    currency = "N/A"
    for pattern, curr in CURRENCY_PATTERNS:
        match = re.search(pattern, clause_text, re.IGNORECASE)
        if match:
            raw_amt = match.group(1).replace(",", "").strip()
            # Ensure raw_amt is not empty and contains at least one digit
            if raw_amt and any(c.isdigit() for c in raw_amt):
                amount = raw_amt
                currency = curr
                break

    deadline = "N/A"
    for dp in DEADLINE_PATTERNS:
        match = re.search(dp, clause_text, re.IGNORECASE)
        if match:
            deadline = clean_entity_text(match.group(1), 12)
            break

    trigger = "N/A"
    for tp in TRIGGER_PATTERNS:
        match = re.search(tp, clause_text, re.IGNORECASE)
        if match:
            trigger = clean_entity_text(match.group(1), 14)
            break

    consequence = "N/A"
    for cp in CONSEQUENCE_PATTERNS:
        match = re.search(cp, clause_text, re.IGNORECASE)
        if match:
            consequence = clean_entity_text(match.group(1), 14)
            break

    refund_cond = "N/A"
    for rp in REFUND_PATTERNS:
        match = re.search(rp, clause_text, re.IGNORECASE)
        if match:
            refund_cond = clean_entity_text(match.group(1), 14)
            break

    # Affected party detection
    party = "consumer"
    if re.search(r'\b(employee|consultant|worker|staff)\b', clause_text, re.IGNORECASE):
        party = "employee"
    elif re.search(r'\b(borrower|mortgagor|debtor)\b', clause_text, re.IGNORECASE):
        party = "borrower"
    elif re.search(r'\b(tenant|lessee|occupant)\b', clause_text, re.IGNORECASE):
        party = "tenant"
    elif re.search(r'\b(distributor|reseller|licensee|vendor|contractor)\b', clause_text, re.IGNORECASE):
        party = "commercial_counterparty"
    elif re.search(r'\b(buyer|allottee|shareholder|founder|investor)\b', clause_text, re.IGNORECASE):
        party = "buyer_or_investor"
    elif re.search(r'\b(customer|subscriber|user|passenger|consumer|client)\b', clause_text, re.IGNORECASE):
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


# ==============================================================================
# SECTION, DATE/TIMELINE & FOCAL RISK PARTS EXTRACTION
# ==============================================================================

FOCAL_RISK_PATTERNS = [
    # 1. renewal subscription / auto renewal
    (r'\b(?:renewal\s+subscription|subscription\s+renewal|recurring\s+(?:annual|monthly)?\s*subscription|automatically\s+renew(?:s|ed|ing)?|auto-renew(?:s|ed|al)?|auto\s+renew(?:s|ed|al)?|renewal\s+term|subsequent\s+renewal)\b', "renewal subscription"),
    # 2. penalty
    (r'\b(?:penalt(?:y|ies)|cancellation\s+fee|exit\s+penalty|late\s+(?:payment\s+)?(?:fee|penalty))\b', "penalty"),
    # 3. notice period
    (r'\b(?:notice\s+period|written\s+notice|prior\s+notice|advance\s+notice|\d+\s*(?:days?|hours?|months?)\s+(?:prior\s+)?notice)\b', "notice period"),
    # 4. hidden fees
    (r'\b(?:hidden\s+fees?|undisclosed\s+(?:fees?|charges?)|additional\s+fees?|processing\s+fee|administrative\s+(?:charge|fee)|restocking\s+fee)\b', "hidden fees"),
    # 5. liability
    (r'\b(?:liabilit(?:y|ies)|limitation\s+of\s+liability|cap\s+on\s+liability|not\s+(?:be\s+held\s+)?liable|no\s+liability|waive(?:s|d)?\s+liability|disclaim(?:s|ed)?\s+all\s+liabilit(?:y|ies))\b', "liability"),
    # 6. liquidity damages / liquidated damages
    (r'\b(?:liquid(?:ity|ated)\s+damages)\b', "liquidity damages"),
    # 7. early termination fees
    (r'\b(?:early\s+termination\s+(?:fees?|charges?)|cancellation\s+fees?|early\s+exit\s+(?:penalty|fees?))\b', "early termination fees"),
    # 8. non cancellable
    (r'\b(?:non[\s\-]cancellable|non[\s\-]refundable|no\s+cash\s+refunds?|no\s+refunds?)\b', "non cancellable"),
    # 9. forfeit
    (r'\b(?:forfeit(?:ure|s|ed|ing)?)\b', "forfeit"),
    # 10. Real Estate / Land
    (r'\b(?:encumbrance|lien|foreclosure|eviction|stamp\s+duty|possession\s+delay|builder\s+delay|carpet\s+area|super\s+built-up|earnest\s+money|cancellation\s+of\s+allotment)\b', "real estate risk"),
    # 11. Loan & Finance
    (r'\b(?:cross-default|penal\s+interest|compound\s+interest|prepayment\s+penalty|acceleration\s+of\s+debt|personal\s+guarantee|loan\s+recall|repossession|collateral\s+forfeiture)\b', "loan risk"),
    # 12. Shareholders & Equity
    (r'\b(?:anti-dilution|liquidation\s+preference|drag-along|tag-along|rofr|vesting\s+cliff|founder\s+lock-up|clawback|squeeze-out|repurchase\s+at\s+nominal)\b', "shareholder equity risk"),
    # 13. IP & Technology
    (r'\b(?:work\s+for\s+hire|ip\s+assignment|proprietary\s+rights|moral\s+rights|perpetual\s+license|patent\s+infringement|copyright\s+indemnity)\b', "intellectual property risk"),
    # Statutory & Unilateral triggers
    (r'\b(?:non-compete|shall\s+not\s+engage|restraint\s+of\s+trade|competing\s+business)\b', "restraint of trade"),
    (r'\b(?:exclusive\s+jurisdiction|waive\s+(?:the\s+)?right\s+to\s+court|cannot\s+sue|binding\s+unilateral\s+arbitration)\b', "court waiver"),
    (r'\b(?:sole\s+discretion|without\s+prior\s+notice|modify\s+at\s+any\s+time)\b', "unilateral modification")
]


def extract_section_info(header: str, clause_text: str = "") -> Dict[str, str]:
    """
    Identifies and parses Section/Clause number and clean title from text header.
    Examples:
      - 'SECTION 1: TICKET FARE RULES & CANCELLATION FEES.' -> number: 'SECTION 1', title: 'Ticket Fare Rules & Cancellation Fees'
      - '1. SUBSCRIPTION TERM & AUTOMATIC RENEWAL.' -> number: 'Section 1', title: 'Subscription Term & Automatic Renewal'
      - 'CLAUSE 2. EARLY TERMINATION CHARGES.' -> number: 'Clause 2', title: 'Early Termination Charges'
    """
    target = header.strip() if header else ""
    if not target or target.startswith("Clause C-"):
        first_line = clause_text.strip().splitlines()[0] if clause_text else ""
        if len(first_line) < 100:
            target = first_line

    section_num = "Section"
    section_title = target

    # Pattern: SECTION 1: ... or Clause 2. ... or Article 3 - ...
    num_match = re.match(r'^(SECTION\s+\d+|CLAUSE\s+\d+|ARTICLE\s+\d+|[0-9]{1,2}\.)[\s:\.\-]+(.*)$', target, re.IGNORECASE)
    if num_match:
        raw_num = num_match.group(1).strip()
        if raw_num.endswith("."):
            section_num = f"Section {raw_num[:-1]}"
        else:
            section_num = raw_num.title()
        section_title = num_match.group(2).strip()
    elif "section" in target.lower():
        section_num = "Section"
    elif "clause" in target.lower():
        section_num = "Clause"

    # Clean punctuation at end of title
    section_title = re.sub(r'[\.:\-]+$', '', section_title).strip()
    if not section_title:
        section_title = target or "Standard Provision"

    return {
        "section_id": section_num,
        "section_title": section_title,
        "full_heading": f"{section_num}: {section_title}" if section_num != section_title else section_title
    }


def extract_dates_and_timelines(text: str) -> List[Dict[str, str]]:
    """
    Extracts explicit dates, notice windows, deadlines, and durations from contract text.
    """
    if not text:
        return []

    timelines = []
    seen = set()

    patterns = [
        # Explicit notice period e.g. at least sixty (60) days prior
        (r'(\b(?:at\s+least\s+)?(?:\w+\s+\(\d+\)|\d+)\s*(?:business\s+)?days?\s+(?:prior|before|advance notice|written notice)\b[^\.,;:]*)', "Notice Window"),
        # Within X business days / hours
        (r'(\bwithin\s+(?:\w+\s+\(\d+\)|\d+)\s*(?:business\s+)?(?:hours?|days?|weeks?|months?)\b[^\.,;:]*)', "Deadline"),
        # Durations e.g. twelve (12) consecutive months, 24-month commitment, 2 years
        (r'(\b(?:for\s+)?(?:twelve\s+\(12\)|\d+|two\s+\(2\))\s*(?:consecutive\s+)?(?:months?|years?|periods?)\b[^\.,;:]*)', "Duration"),
        # Specific recurrence e.g. 1st of each month, annually, recurring
        (r'(\b(?:1st|2nd|3rd|\d{1,2}th)\s+of\s+each\s+month\b)', "Payment Schedule"),
        # Effective Date or Commencement
        (r'(\b(?:Effective\s+Date|Commencement\s+Date|expiration\s+of\s+(?:the\s+)?(?:current\s+)?term)\b)', "Milestone Date"),
        # Hours notice e.g. 24 hours notice, 72 hours
        (r'(\b\d+\s*hours?\s+(?:notice|prior|departure)?[^\.,;:]*)', "Notice Window")
    ]

    for pat, time_type in patterns:
        for match in re.finditer(pat, text, re.IGNORECASE):
            val = match.group(1).strip()
            val_clean = re.sub(r'\s+', ' ', val)
            if len(val_clean) >= 3 and val_clean.lower() not in seen:
                seen.add(val_clean.lower())
                timelines.append({
                    "type": time_type,
                    "value": val_clean
                })

    return timelines


def split_clause_sentences(text: str) -> List[str]:
    """Splits a clause into grammatical sentences while protecting legal abbreviations."""
    if not text:
        return []
    # Replace newlines with spaces
    cleaned = re.sub(r'\r\n|\r|\n', ' ', text)
    # Split on periods followed by whitespace and capital letter or number, while avoiding Rs. or Dr.
    sentences = re.split(r'(?<!\bRs)(?<!\bNo)(?<!\be\.g)(?<!\bi\.e)(?<!\bvs)(?<!\bSec)(?<!\bArt)\.\s+(?=[A-Z0-9\(\[])', cleaned)
    res = []
    for s in sentences:
        s_strip = s.strip()
        if s_strip and not s_strip.endswith('.'):
            s_strip += '.'
        if len(s_strip) >= 12:
            res.append(s_strip)
    return res if res else [cleaned]


def extract_focal_risk_parts(
    clause_text: str,
    highlight_phrases: Optional[List[str]] = None,
    entities: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extracts only the critical focal risk parts of a clause rather than the entire text.
    Identifies exact trigger sentences matching core focal terms (renewal subscription, penalty,
    notice period, hidden fees, liability, liquidity damages, early termination fees, non cancellable, forfeit).
    """
    if not clause_text:
        return {
            "main_parts": [],
            "primary_focus_text": "",
            "focal_keywords_matched": [],
            "is_shortened": False
        }

    sentences = split_clause_sentences(clause_text)
    focal_parts = []
    matched_focal_labels = set()
    all_matched_tokens = []

    for s in sentences:
        s_lower = s.lower()
        matched_in_sentence = []

        # 1. Check FOCAL_RISK_PATTERNS
        for pat_str, label in FOCAL_RISK_PATTERNS:
            for m in re.finditer(pat_str, s, re.IGNORECASE):
                token = m.group(0)
                matched_in_sentence.append(token)
                matched_focal_labels.add(label)
                all_matched_tokens.append(token)

        # 2. Check highlight phrases passed in from risk gating
        if highlight_phrases:
            for phrase in highlight_phrases:
                if phrase and phrase.lower() in s_lower:
                    matched_in_sentence.append(phrase)
                    all_matched_tokens.append(phrase)

        # 3. Check financial entities present in this sentence
        if entities:
            amt = str(entities.get("amount", ""))
            if amt and amt != "N/A" and amt in s:
                matched_in_sentence.append(f"{entities.get('currency', '₹')}{amt}")

        if matched_in_sentence:
            focal_parts.append({
                "sentence": s.strip(),
                "focal_triggers": list(dict.fromkeys(matched_in_sentence)),
                "trigger_count": len(matched_in_sentence)
            })

    # Sort focal parts by trigger density
    focal_parts.sort(key=lambda x: x["trigger_count"], reverse=True)

    # If no sentences matched specific focal patterns, take first informative sentence
    if not focal_parts:
        primary_focus_text = sentences[0] if sentences else clause_text[:180]
        main_parts_texts = [primary_focus_text]
        is_shortened = len(clause_text) > len(primary_focus_text)
    else:
        # Keep top 1 to 3 focal parts that capture all the core risks
        selected = focal_parts[:2]
        main_parts_texts = [p["sentence"] for p in selected]
        primary_focus_text = " ".join(main_parts_texts)
        is_shortened = len(clause_text.strip()) > (len(primary_focus_text.strip()) + 20)

    return {
        "main_parts": main_parts_texts,
        "primary_focus_text": primary_focus_text,
        "focal_keywords_matched": sorted(list(matched_focal_labels)),
        "matched_tokens": list(dict.fromkeys(all_matched_tokens)),
        "is_shortened": is_shortened
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
