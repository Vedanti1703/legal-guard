"""
server.py

Web Server for NLP Consumer T&C & Contract Risk Analyzer.
Provides REST API endpoints for document analysis, file upload, preset samples, and semantic vector search.
Serves the rich modern Web Frontend on http://127.0.0.1:5000
"""

import os
import sys
import json
import logging
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.common import setup_logger, clean_legal_text
from src.inference.analyzer import analyze_contract_text, semantic_search_clauses, get_models

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


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    models = get_models()
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "tfidf": models.get("tfidf") is not None,
            "legal_bert": models.get("bert_model") is not None,
            "sentence_bert": models.get("sbert") is not None,
            "acord_vector_index": models.get("acord_embeddings") is not None
        }
    })


@app.route("/api/samples", methods=["GET"])
def get_samples():
    return jsonify({
        "samples": PRESET_SAMPLES
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    contract_text = ""

    # Check if multipart form upload with file
    if "file" in request.files:
        uploaded_file = request.files["file"]
        filename = uploaded_file.filename.lower()
        
        try:
            if filename.endswith(".pdf"):
                # Use pypdfium2 or pdfplumber to extract text
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(uploaded_file)
                extracted_pages = []
                for page in pdf:
                    text_page = page.get_textpage()
                    extracted_pages.append(text_page.get_text_range())
                contract_text = "\n\n".join(extracted_pages)
            elif filename.endswith(".txt") or filename.endswith(".json"):
                contract_text = uploaded_file.read().decode("utf-8", errors="ignore")
            else:
                return jsonify({"error": "Unsupported file format. Please upload a .pdf, .txt, or .json file."}), 400
        except Exception as e:
            logger.error(f"Error reading uploaded file: {e}")
            return jsonify({"error": f"Failed to parse uploaded document: {str(e)}"}), 500
    else:
        # JSON payload text
        data = request.get_json(silent=True) or {}
        contract_text = data.get("text", "")

    contract_text = clean_legal_text(contract_text)
    if not contract_text or len(contract_text) < 15:
        return jsonify({"error": "Please provide valid contract or T&C text for analysis (at least 15 characters)."}), 400

    try:
        results = analyze_contract_text(contract_text)
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error during contract analysis: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred during NLP analysis: {str(e)}"}), 500


@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json() or {}
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
    # Pre-warm models on startup
    get_models()
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Consumer T&C Analyzer Web App on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
