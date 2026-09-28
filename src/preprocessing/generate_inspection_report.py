"""
src/preprocessing/generate_inspection_report.py

Runs inspections across CUAD, ACORD, and the Indian dataset,
and writes the comprehensive report to outputs/reports/dataset_inspection.txt.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.preprocessing.inspect_cuad import inspect_cuad, format_cuad_report
from src.preprocessing.inspect_acord import inspect_acord, format_acord_report
from src.preprocessing.inspect_indian import inspect_indian, format_indian_report
from src.utils.common import get_project_root, setup_logger

logger = setup_logger("generate_inspection_report")


def generate_full_report() -> Path:
    root = get_project_root()
    output_path = root / "outputs" / "reports" / "dataset_inspection.txt"
    os.makedirs(output_path.parent, exist_ok=True)

    logger.info("Running CUAD inspection...")
    cuad_res = inspect_cuad()
    cuad_text = format_cuad_report(cuad_res)

    logger.info("Running ACORD inspection...")
    acord_res = inspect_acord()
    acord_text = format_acord_report(acord_res)

    logger.info("Running Indian dataset inspection...")
    indian_res = inspect_indian()
    indian_text = format_indian_report(indian_res)

    lines = []
    lines.append("=" * 80)
    lines.append("BTECH NLP PROJECT: DATASET INSPECTION & RECONNAISSANCE REPORT")
    lines.append("Project: NLP for Detecting Hidden Fees and Unfavorable Terms in Consumer T&Cs")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("\n\n")

    lines.append(cuad_text)
    lines.append("\n\n")

    lines.append(acord_text)
    lines.append("\n\n")

    lines.append(indian_text)
    lines.append("\n\n")

    lines.append("=" * 80)
    lines.append("ARCHITECTURAL SYNTHESIS: PURPOSE & PIPELINE ROLE FOR EACH DATASET")
    lines.append("=" * 80)
    lines.append("1. CUAD (Contract Understanding Atticus Dataset):")
    lines.append("   - Primary Task: Legal Clause Understanding & Multi-Class Classification.")
    lines.append("   - 510 contracts, 13,823 labeled spans across 41 categories.")
    lines.append("   - Includes target consumer-risk categories: Renewal Term (210), Notice Period (122),")
    lines.append("     Termination For Convenience (246), Liquidated Damages (121), Post-Termination Services (450),")
    lines.append("     and Revenue/Profit Sharing (418).")
    lines.append("   - Crucial constraint: Split must be strictly document-level (contract ID based) to avoid leakage.")
    lines.append("")
    lines.append("2. ACORD (Atticus Clause Retrieval Dataset):")
    lines.append("   - Primary Task: Semantic Retrieval and Ranking.")
    lines.append("   - Structure: 114 queries, 3,931 corpus clauses, 126,659 graded pairs (0 to 4 rating).")
    lines.append("   - Evaluation: Recall@1, Recall@5, Recall@10, and Mean Reciprocal Rank (MRR).")
    lines.append("   - Not a classification dataset; tests semantic search for consumer clauses.")
    lines.append("")
    lines.append("3. Indian Legal Dataset:")
    lines.append("   - Primary Task: Domain Adaptation, Risky Pattern Rule Extraction, and Statutory Grounding.")
    lines.append("   - Structure: 5 PDF files (136 pages, 100% text extraction success).")
    lines.append("   - Contains: Indian Contract Act 1872 statutory text + 12 categories of risky contract patterns")
    lines.append("     and curated Good vs. Bad Indian contract clauses.")
    lines.append("   - Provides legal grounding under Indian jurisprudence (e.g. Sec 27 voiding non-competes).")
    lines.append("=" * 80)

    full_report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_report_text)

    logger.info(f"Report written successfully to: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_full_report()
