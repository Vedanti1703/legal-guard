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
    "loan_agreement": {
        "title": "Term Loan Agreement",
        "text": """LOAN AGREEMENT

THIS LOAN AGREEMENT ("Agreement") is entered into as of the 1st day of January, 2024, between ABC Lending Corporation, a company incorporated under the Companies Act, 2013, having its registered office at Mumbai, Maharashtra ("Lender"), and XYZ Enterprises Pvt. Ltd., having its registered office at Bangalore, Karnataka ("Borrower").

1. LOAN AMOUNT & DISBURSEMENT.
The Lender agrees to advance a term loan of INR 50,00,000 (Rupees Fifty Lakhs only) ("Loan Amount") to the Borrower, subject to the terms herein. Disbursement shall be made within 5 business days of execution of this Agreement and satisfaction of all conditions precedent including submission of audited financial statements and execution of security documents.

2. INTEREST RATE & COMPUTATION.
The Loan shall carry interest at the rate of 18% per annum, computed on a monthly reducing balance basis. Interest shall accrue from the date of disbursement. In the event of any default in payment, penal interest at the rate of 24% per annum shall be levied on the overdue amount from the date of default until the date of actual payment.

3. REPAYMENT SCHEDULE & PREPAYMENT PENALTY.
The Loan shall be repayable in 36 equated monthly installments (EMIs) of INR 1,80,760 each, commencing on the 1st of the month following disbursement. Any prepayment of the Loan, whether in full or in part, shall attract a prepayment penalty of 3% of the prepaid amount. No prepayment shall be accepted without 30 days prior written notice to the Lender.

4. SECURITY & COLLATERAL.
As security for the Loan, the Borrower shall create a first and exclusive charge over all movable and immovable assets of the Company, both present and future. The Borrower shall also furnish a personal guarantee of all directors of the Company in a form acceptable to the Lender.

5. EVENTS OF DEFAULT & ACCELERATION.
The following shall constitute events of default: (a) failure to pay any installment within 5 days of due date; (b) any material adverse change in the financial condition of the Borrower; (c) commencement of insolvency proceedings; (d) breach of any covenant herein. Upon occurrence of an Event of Default, the entire outstanding Loan Amount together with accrued interest, penal interest, and all charges shall become immediately due and payable without notice.

6. UNILATERAL AMENDMENT.
The Lender reserves the right to revise interest rates, charges, and repayment terms at its sole discretion upon 7 days written notice to the Borrower. Continued acceptance of the Loan shall constitute acceptance of revised terms.

7. JURISDICTION.
All disputes arising under this Agreement shall be subject to the exclusive jurisdiction of the courts at Mumbai, Maharashtra."""
    },
    "shareholders_agreement": {
        "title": "Shareholders Agreement",
        "text": """SHAREHOLDERS AGREEMENT

THIS SHAREHOLDERS AGREEMENT ("Agreement") is made and entered into as of 15th March 2024 among TechVenture Innovations Pvt. Ltd. (the "Company"), Alpha Ventures Fund II LLP ("Investor"), and the Founders listed in Schedule A collectively ("Founders").

1. SHARE CAPITAL & EQUITY STRUCTURE.
The authorized share capital of the Company is INR 10,00,000 divided into 10,00,000 equity shares of INR 1 each. The Investor shall subscribe to 25% of the fully diluted share capital at a pre-money valuation of INR 4,00,00,000. The Founders collectively hold the remaining 75% equity subject to a 4-year vesting schedule with a 1-year cliff.

2. FOUNDER VESTING & REVERSE VESTING.
Each Founder's shares shall be subject to reverse vesting over a period of 48 months with a 12-month cliff. In the event any Founder ceases to be a full-time employee or director of the Company ("Bad Leaver"), all unvested shares shall be compulsorily transferred to the Company at par value of INR 1 per share, irrespective of the prevailing fair market value.

3. ANTI-DILUTION PROTECTION.
In the event the Company issues any securities at a valuation lower than the Investor's entry valuation ("Down Round"), the Investor shall be entitled to receive additional shares from the Founders' pool on a full ratchet anti-dilution basis to maintain the Investor's economic ownership percentage at no additional cost.

4. DRAG-ALONG RIGHTS.
If shareholders holding more than 75% of the fully diluted share capital approve a sale or merger of the Company, all remaining shareholders including Founders shall be compelled to sell their shares at the same price and on the same terms as approved by the majority. Founders shall have no right to block or delay such a transaction.

5. RIGHT OF FIRST REFUSAL & CO-SALE.
Before any Founder may transfer shares to any third party, the Investor shall have a Right of First Refusal to purchase such shares at the same price. If the Investor declines, it shall have a Co-Sale right to sell an equivalent proportion of its shares alongside the Founder at the same terms. Any transfer in violation of this clause shall be null and void.

6. RESERVED MATTERS.
The following actions shall require prior written consent of the Investor: (a) amendment of Articles of Association; (b) issue of any new securities; (c) incurring debt exceeding INR 25 lakhs; (d) disposal of material assets; (e) change in the nature of business; (f) appointment or removal of key managerial personnel.

7. LIQUIDATION PREFERENCE.
Upon a liquidation, dissolution, or deemed liquidation event (including acquisition), the Investor shall be entitled to receive, in preference to all other shareholders, an amount equal to 1.5x the investment amount plus any declared but unpaid dividends before any distribution is made to the Founders or other shareholders."""
    },
    "real_estate_agreement": {
        "title": "Real Estate Sale Agreement",
        "text": """AGREEMENT FOR SALE OF IMMOVABLE PROPERTY

THIS AGREEMENT FOR SALE ("Agreement") is executed on this 10th day of February, 2024, at Pune, Maharashtra, between Mr. Ramesh Sharma, son of Mr. Suresh Sharma, resident of Pune ("Seller"), and Mrs. Priya Mehta, daughter of Mr. Anil Kumar, resident of Mumbai ("Buyer").

1. PROPERTY DESCRIPTION.
The Seller agrees to sell and the Buyer agrees to purchase the residential flat bearing Flat No. 402, 4th Floor, Sai Residency, Survey No. 45/2, Baner, Pune – 411045, admeasuring 1,150 sq.ft. of carpet area ("Property"), free from all encumbrances, liens, mortgages, and pending litigation.

2. SALE CONSIDERATION & PAYMENT SCHEDULE.
The total sale consideration for the Property is INR 78,00,000 (Rupees Seventy-Eight Lakhs only). The Buyer shall pay as follows: (a) Advance/Token Amount: INR 5,00,000 on signing this Agreement; (b) Second Installment: INR 20,00,000 within 45 days; (c) Balance: INR 53,00,000 on or before the date of registration.

3. FORFEITURE OF ADVANCE ON DEFAULT.
In the event the Buyer fails to complete the purchase within the stipulated timeframe or defaults on any payment schedule, the Seller shall be entitled to forfeit the entire advance amount paid as liquidated damages. The Agreement shall stand terminated without further notice and the Seller shall be free to deal with the Property as it deems fit.

4. PENALTY ON SELLER'S DEFAULT.
In the event the Seller fails to execute the sale deed or creates any encumbrance on the Property, the Seller shall refund double the advance amount (INR 10,00,000) as penalty to the Buyer within 30 days.

5. TITLE & ENCUMBRANCES.
The Seller warrants that the Property has a clear and marketable title and is not subject to any government acquisition notice, court order, or pending litigation. The Seller shall provide all original title documents for inspection by the Buyer's advocate prior to execution of the sale deed.

6. POSSESSION & REGISTRATION.
Possession of the Property shall be handed over to the Buyer on the date of registration of the Sale Deed. Registration shall be completed within 60 days from the date of this Agreement at the Sub-Registrar's Office, Pune. All stamp duty and registration charges shall be borne exclusively by the Buyer.

7. BROKERAGE & INCIDENTAL COSTS.
Brokerage commission of 1% of the sale consideration shall be jointly borne by both parties. All costs of obtaining NOC from the Housing Society shall be borne by the Seller."""
    },
    "land_lease_agreement": {
        "title": "Land Lease Agreement",
        "text": """LAND LEASE AGREEMENT

THIS LAND LEASE AGREEMENT ("Agreement") is entered into on this 5th day of April, 2024 between Mr. Sunil Patil, adult, residing at Nashik, Maharashtra ("Lessor/Landowner"), and GreenField Agro Industries Pvt. Ltd., a company incorporated under the Companies Act, 2013, having its registered office at Pune, Maharashtra ("Lessee").

1. DESCRIPTION OF LEASED LAND.
The Lessor hereby leases to the Lessee agricultural land admeasuring 12 acres (approximately 48,562 sq. meters) situated at Survey No. 78, Village Ozar, Taluka Niphad, District Nashik, Maharashtra ("Leased Land"), for the purposes of agricultural activities and agro-processing only.

2. LEASE TERM & RENEWAL.
The lease shall be for an initial period of 5 (Five) years commencing from 1st May 2024 to 30th April 2029. The Lessee shall have the option to renew the lease for one further period of 5 years on the same terms, provided a renewal notice is given at least 90 days prior to expiry. Failure to provide timely notice shall result in automatic termination of lease rights without compensation.

3. LEASE RENT & ESCALATION.
The Lessee shall pay an annual lease rent of INR 3,60,000 (Rupees Three Lakhs Sixty Thousand only) payable in advance on 1st May of each year. The lease rent shall be subject to an annual escalation of 10% on the prevailing rent at the commencement of each lease year. Failure to pay rent within 15 days of due date shall attract a penalty of 2% per month on the outstanding amount.

4. SECURITY DEPOSIT.
The Lessee shall deposit an interest-free security deposit of INR 5,00,000 with the Lessor at the time of execution of this Agreement, which shall be refunded within 60 days of expiry or termination of the Lease, subject to deductions for any damage to the land, crops, or structures caused by the Lessee.

5. PERMITTED USE & RESTRICTIONS.
The Lessee shall use the Leased Land exclusively for agricultural cultivation and agro-processing activities. The Lessee shall not: (a) sublet or sub-license the land to any third party; (b) construct any permanent structures without prior written consent of the Lessor; (c) fell trees or alter the natural topography of the land; (d) use the land for any non-agricultural commercial purpose.

6. EARLY TERMINATION.
The Lessor shall have the right to terminate this Agreement prior to expiry upon 60 days written notice in the event of: (a) non-payment of rent for 2 consecutive months; (b) breach of permitted use conditions; (c) insolvency of the Lessee. Upon early termination, the Lessee shall vacate the land within 30 days and forfeit any improvements made to the land. No compensation shall be payable to the Lessee for standing crops or investments made.

7. DISPUTE RESOLUTION.
Any dispute arising from this Agreement shall be referred to arbitration under the Arbitration and Conciliation Act, 1996. The arbitration shall be conducted by a sole arbitrator mutually appointed by the parties, failing which the arbitrator shall be appointed by the District Court, Nashik. The seat of arbitration shall be Nashik, Maharashtra."""
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
