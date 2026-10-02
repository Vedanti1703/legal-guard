"""
server.py

Web Server for NLP Consumer T&C & Contract Risk Analyzer.
Provides REST API endpoints for:
- Document analysis with Two-Stage Gating (/api/analyze)
- Document Summarization with Extractive/Abstractive/Hybrid modes (/api/summarize)
- Multi-Model Registry & Metrics comparison (/api/models)
- Preset samples and Semantic Vector Search (/api/samples, /api/search)
Serves the rich modern Web Frontend on http://127.0.0.1:5000
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Tuple, Optional
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.common import setup_logger, clean_legal_text
from src.inference.analyzer import (
    analyze_contract_text,
    semantic_search_clauses,
    get_models,
    get_available_models_info
)
from src.summarization.summarizer import summarize_document

logger = setup_logger("web_server")

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# Pre-defined real-world sample contracts for instant 1-click testing
PRESET_SAMPLES = {
    "saas_autorenewal": {
        "title": "SaaS Subscription Agreement (Hidden Renewal & Penalty)",
        "text": """1. SUBSCRIPTION TERM & AUTOMATIC RENEWAL.
The Initial Subscription Term shall commence on the Effective Date and continue for twelve (12) consecutive months. UNLESS CANCELLED BY WRITTEN NOTICE AT LEAST SIXTY (60) DAYS PRIOR TO THE EXPIRATION OF THE CURRENT TERM, THIS AGREEMENT SHALL AUTOMATICALLY RENEW FOR ADDITIONAL 12-MONTH PERIODS. A recurring annual subscription charge of ₹24,999 will be automatically debited from the Customer's registered payment method without prior invoice notification.

2. CANCELLATION PENALTY & FORFEITURE.
If the Customer terminates this Agreement prior to the expiration of the subscription period, the Customer shall incur a cancellation fee of Rs 10,000 as liquidated damages within 7 business days. All accrued fees and advance payments shall be non-refundable. No cash refunds or pro-rated credits will be issued under any circumstances.

3. RESTRAINT OF PROFESSION & NON-COMPETE.
During the subscription term and for a period of two (2) years following any termination hereof, the Customer and its affiliates shall not engage in, develop, or operate any software product or business that competes with the Company's services within the territory of India.

4. UNILATERAL PRICE ADJUSTMENT.
The Company reserves the right to modify, increase, or adjust subscription pricing, service tiers, or feature access at any time at its sole discretion upon 24 hours notice published on the website.

5. RESTRICTION OF LEGAL RECOURSE & EXCLUSIVE JURISDICTION.
The Customer hereby waives any right to file suit in any public court or consumer disputes forum. All disputes arising out of this contract shall be submitted to binding unilateral arbitration conducted by a sole arbitrator appointed exclusively by the Company in London, UK."""
    },
    "airline_cancellation": {
        "title": "Airline Passenger Ticket & Refund Policy",
        "text": """SECTION 1: TICKET FARE RULES & CANCELLATION FEES.
All promotional economy fares purchased through our online booking engine are strictly non-refundable. Cancellations requested within 72 hours of scheduled flight departure will incur a cancellation fee of ₹4,500 per passenger per segment.

SECTION 2: SCHEDULE CHANGES & DELAYS.
In the event of flight delay, cancellation, or schedule modification by the carrier exceeding 6 hours, the passenger's sole remedy shall be a rebooking on the next available flight. No cash refund or hotel accommodation reimbursement shall be provided.

SECTION 3: LUGGAGE LOSS & LIMITATION OF LIABILITY.
In no event shall the Carrier's liability for lost, damaged, or delayed baggage exceed $100 USD total per passenger, regardless of actual loss incurred or declared value."""
    },
    "gym_membership": {
        "title": "Fitness Club Membership & Penalty Contract",
        "text": """CLAUSE 1. MEMBERSHIP DUES & AUTOMATIC DEBIT.
Membership requires a minimum 24-month commitment. Dues of Rs 2,999 per month will be charged via auto-debit on the 1st of each month. 

CLAUSE 2. EARLY TERMINATION CHARGES.
If the member terminates before completing 24 months, a mandatory early exit penalty of ₹12,000 shall be levied immediately. Failure to pay within 14 days will attract a late interest charge of 18% per annum.

CLAUSE 3. WAIVER OF INJURY LIABILITY.
The gym facility shall not be held liable for any personal injury, permanent disability, or property loss suffered by the member while using fitness equipment, regardless of equipment maintenance or staff negligence."""
    },
    "ecommerce_return": {
        "title": "E-Commerce Consumer Purchase Terms",
        "text": """1. RETURN & REFUND RESTRICTIONS.
Item returns must be initiated within 3 days of delivery. Returns will be refunded strictly in the form of non-transferable store reward points. No bank refunds or cash returns will be processed.

2. RESTOCKING FEE.
All returned electronics items are subject to a 20% restocking fee deducted automatically from the credit balance.

3. DAMAGE IN TRANSIT.
Claims for damaged items must be accompanied by an unboxing video recorded continuously without cuts. Reports submitted after 24 hours of delivery shall be rejected without review."""
    }
}


def extract_text_from_upload(req) -> Tuple[str, Optional[str]]:
    """Helper to extract text from request (file upload or JSON/form body)."""
    if "file" in req.files:
        uploaded_file = req.files["file"]
        filename = uploaded_file.filename.lower()
        try:
            if filename.endswith(".pdf"):
                try:
                    import pypdfium2 as pdfium
                    pdf = pdfium.PdfDocument(uploaded_file)
                    extracted_pages = []
                    for page in pdf:
                        text_page = page.get_textpage()
                        extracted_pages.append(text_page.get_text_range())
                    return "\n\n".join(extracted_pages), None
                except Exception:
                    import fitz  # PyMuPDF fallback
                    uploaded_file.seek(0)
                    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                    text = "\n\n".join([page.get_text() for page in doc])
                    return text, None
            elif filename.endswith(".txt") or filename.endswith(".json"):
                return uploaded_file.read().decode("utf-8", errors="ignore"), None
            else:
                return "", "Unsupported file format. Please upload a .pdf, .txt, or .json file."
        except Exception as e:
            return "", f"Failed to parse uploaded document: {str(e)}"
    else:
        data = req.get_json(silent=True) or {}
        text = data.get("text") or req.form.get("text", "")
        return text, None


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    models = get_models()
    return jsonify({
        "status": "healthy",
        "active_model": models.get("active_model", "legalbert"),
        "models_loaded": {
            "tfidf": models.get("tfidf") is not None,
            "legal_bert": models.get("bert_model") is not None,
            "deberta": models.get("deberta_model") is not None,
            "sentence_bert": models.get("sbert") is not None,
            "acord_vector_index": models.get("acord_embeddings") is not None
        }
    })


@app.route("/api/models", methods=["GET"])
def get_models_endpoint():
    """Returns loaded models and their benchmark test metrics."""
    try:
        info = get_available_models_info()
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error fetching models info: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/samples", methods=["GET"])
def get_samples():
    return jsonify({"samples": PRESET_SAMPLES})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    contract_text, err = extract_text_from_upload(request)
    if err:
        return jsonify({"error": err}), 400

    contract_text = clean_legal_text(contract_text)
    if not contract_text or len(contract_text) < 15:
        return jsonify({"error": "Please provide valid contract or T&C text for analysis (at least 15 characters)."}), 400

    # Query params / body params for output gating
    data = request.get_json(silent=True) or {}
    include_safe_raw = request.args.get("include_safe", data.get("include_safe", request.form.get("include_safe", "false")))
    include_safe = str(include_safe_raw).lower() in ["true", "1", "yes"]

    min_risk_level = request.args.get("min_risk_level", data.get("min_risk_level", request.form.get("min_risk_level", "MEDIUM")))

    try:
        results = analyze_contract_text(
            full_text=contract_text,
            min_risk_level=min_risk_level,
            include_safe=include_safe
        )
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error during contract analysis: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred during NLP analysis: {str(e)}"}), 500


@app.route("/api/summarize", methods=["POST"])
def summarize():
    """Standalone endpoint for whole-document contract summarization."""
    contract_text, err = extract_text_from_upload(request)
    if err:
        return jsonify({"error": err}), 400

    data = request.get_json(silent=True) or {}
    mode = request.args.get("mode", data.get("mode", request.form.get("mode", "hybrid")))
    length = request.args.get("length", data.get("length", request.form.get("length", "medium")))

    try:
        summary_result = summarize_document(
            text=contract_text,
            mode=mode,
            length=length
        )
        if "error" in summary_result:
            return jsonify(summary_result), 400
        return jsonify(summary_result)
    except Exception as e:
        logger.error(f"Error during document summarization: {e}", exc_info=True)
        return jsonify({"error": f"Summarization failed: {str(e)}"}), 500


@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    target_clauses = data.get("clauses", [])

    if not query:
        return jsonify({"error": "Search query cannot be empty."}), 400

    try:
        results = semantic_search_clauses(query, target_clauses=target_clauses, top_k=5)
        return jsonify({"query": query, "results": results})
    except Exception as e:
        logger.error(f"Error during semantic search: {e}")
        return jsonify({"error": f"Semantic search failed: {str(e)}"}), 500


if __name__ == "__main__":
    get_models()
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Consumer T&C Analyzer Web App on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
