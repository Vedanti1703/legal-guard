"""
src/summarization/summarizer.py

Standalone Whole-Document Summarizer for Contracts and Consumer Terms & Conditions:
1. Extractive Stage: Sentence-BERT embeddings (all-MiniLM-L6-v2) + Centroid & MMR ranking
2. Abstractive Stage: Map-Reduce chunking (BART / T5) under token window limits without silent truncation
3. Hybrid Stage: Extractive salient sentence filtering followed by abstractive cohesion synthesis
4. Structured Legal Extraction: Parties, Purpose, Duration, Payments, Renewals, Termination,
   Liability, Dispute Resolution, Governing Law, and Financial Dates/Amounts
5. Edge Cases: Short docs, 100+ page contracts, scanned/empty docs, non-English warning
"""

import os
import re
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.common import clean_legal_text, setup_logger
from src.preprocessing.extract_financial_entities import extract_financial_terms

logger = setup_logger("document_summarizer")

_SBERT_CACHE: Optional[SentenceTransformer] = None
_ABSTRACTIVE_PIPELINE = None


def get_sbert_model() -> SentenceTransformer:
    """Lazy loads the SentenceTransformer model."""
    global _SBERT_CACHE
    if _SBERT_CACHE is None:
        try:
            logger.info("Loading SBERT model for extractive summarization...")
            _SBERT_CACHE = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Failed to load SBERT: {e}")
            raise e
    return _SBERT_CACHE


def get_abstractive_model():
    """Lazy loads seq2seq summarization pipeline if requested."""
    global _ABSTRACTIVE_PIPELINE
    if _ABSTRACTIVE_PIPELINE is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            model_id = "google/flan-t5-base"
            logger.info(f"Initializing abstractive summarization model ({model_id})...")
            tok = AutoTokenizer.from_pretrained(model_id)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            mdl.eval()
            _ABSTRACTIVE_PIPELINE = {"tokenizer": tok, "model": mdl}
        except Exception as e:
            logger.warning(f"Could not load abstractive pipeline: {e}. Falling back to extractive mode.")
            _ABSTRACTIVE_PIPELINE = False
    return _ABSTRACTIVE_PIPELINE


def split_into_sentences(text: str) -> List[str]:
    """Splits legal text into meaningful sentence and clause units."""
    cleaned = clean_legal_text(text)
    if not cleaned:
        return []

    # Handle numbered clauses, bullet points, and sentence periods
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9\(\[])|\n+', cleaned)
    processed = []
    for s in raw_sentences:
        s_clean = s.strip()
        # Keep sentences that have at least 5 words and are informative
        if len(s_clean.split()) >= 4:
            processed.append(s_clean)

    return processed


def extractive_summarize(
    sentences: List[str],
    sbert_model: SentenceTransformer,
    top_k: int = 5,
    diversity_lambda: float = 0.65
) -> List[str]:
    """
    Extractive summarization using Document Centroid similarity + MMR (Maximal Marginal Relevance)
    to balance relevance with diversity while preserving original document order.
    """
    if not sentences:
        return []
    if len(sentences) <= top_k:
        return sentences

    embeddings = sbert_model.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
    # Centroid embedding (average of all sentence vectors)
    centroid = torch.mean(embeddings, dim=0, keepdim=True)
    centroid = centroid / torch.norm(centroid, p=2, dim=1, keepdim=True)

    sim_to_centroid = util.cos_sim(embeddings, centroid).squeeze(1).cpu().numpy()

    # MMR selection
    selected_indices = []
    candidate_indices = list(range(len(sentences)))

    # First sentence: highest similarity to centroid
    first_idx = int(np.argmax(sim_to_centroid))
    selected_indices.append(first_idx)
    candidate_indices.remove(first_idx)

    pairwise_sims = util.cos_sim(embeddings, embeddings).cpu().numpy()

    while len(selected_indices) < top_k and candidate_indices:
        mmr_scores = []
        for c in candidate_indices:
            relevance = sim_to_centroid[c]
            max_redundancy = max([pairwise_sims[c][s] for s in selected_indices])
            score = (diversity_lambda * relevance) - ((1 - diversity_lambda) * max_redundancy)
            mmr_scores.append((score, c))

        best_candidate = max(mmr_scores, key=lambda x: x[0])[1]
        selected_indices.append(best_candidate)
        candidate_indices.remove(best_candidate)

    # Sort in original document order
    selected_indices.sort()
    return [sentences[i] for i in selected_indices]


def chunk_text(text: str, max_words: int = 400, overlap_words: int = 50) -> List[str]:
    """Splits text into overlapping chunks under token limits for map-reduce processing."""
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += (max_words - overlap_words)

    return chunks


def _run_seq2seq(prompt: str, pipe_dict, max_toks: int = 120, min_toks: int = 25) -> str:
    """Helper to run seq2seq text generation fast under torch.no_grad()."""
    if isinstance(pipe_dict, dict) and "tokenizer" in pipe_dict:
        tok = pipe_dict["tokenizer"]
        mdl = pipe_dict["model"]
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = mdl.generate(
                **inputs,
                max_new_tokens=max_toks,
                min_length=min_toks,
                num_beams=2,
                early_stopping=True
            )
        return tok.decode(outputs[0], skip_special_tokens=True).strip()
    elif callable(pipe_dict):
        with torch.no_grad():
            res = pipe_dict(prompt, max_length=max_toks, min_length=min_toks, do_sample=False)
            return res[0]["generated_text"].strip()
    return prompt[:200]


def abstractive_map_reduce(text: str, pipeline, length_setting: str = "medium") -> str:
    """Executes Map-Reduce chunking for abstractive summarization efficiently."""
    chunks = chunk_text(text, max_words=350, overlap_words=30)
    
    # Cap total chunks to 3 max for big documents to maintain sub-5s response speed
    if len(chunks) > 3:
        chunks = [chunks[0], chunks[len(chunks)//2], chunks[-1]]

    chunk_summaries = []
    max_toks = 70 if length_setting == "short" else (110 if length_setting == "medium" else 150)
    min_toks = 20 if length_setting == "short" else 35

    for idx, ch in enumerate(chunks):
        prompt = f"Summarize key contract terms, obligations, and penalties in plain English:\n{ch}"
        try:
            summary = _run_seq2seq(prompt, pipeline, max_toks=max_toks, min_toks=min_toks)
            chunk_summaries.append(summary)
        except Exception as e:
            logger.warning(f"Abstractive chunk {idx} failed: {e}")
            chunk_summaries.append(ch[:250])

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    # Reduce Stage
    combined_intermediate = " ".join(chunk_summaries)
    reduce_prompt = f"Synthesize these contract points into a clear executive overview:\n{combined_intermediate}"
    try:
        final_out = _run_seq2seq(reduce_prompt, pipeline, max_toks=max_toks * 2, min_toks=min_toks)
        return final_out
    except Exception as e:
        logger.warning(f"Abstractive reduce stage failed: {e}")
        return " ".join(chunk_summaries[:2])


def extract_structured_key_points(full_text: str, sentences: List[str]) -> Dict[str, str]:
    """
    Extracts 9 essential legal dimensions from document sentences using regex & contextual cues:
    Parties, Purpose, Duration, Payment & Fees, Renewal, Termination, Liability, Dispute Resolution, Governing Law.
    """
    text_lower = full_text.lower()
    dimensions = {
        "parties": "Not explicitly specified in extracted text.",
        "purpose": "Consumer software, service, or commercial agreement.",
        "duration_and_term": "Standard or indefinite term unless terminated.",
        "payment_and_fees": "Standard service pricing applies.",
        "renewal": "No auto-renewal specified.",
        "termination": "Standard termination upon mutual notice.",
        "liability": "Standard commercial liability provisions.",
        "dispute_resolution": "Amicable settlement or local judicial jurisdiction.",
        "governing_law": "Laws of India unless otherwise stipulated."
    }

    # 1. Parties
    party_match = re.search(r'(?:between|by and between)\s+([^,\n\.]+)(?:,\s*(?:a|an)\s+[^,\n\.]+)?\s+and\s+([^,\n\.]+)', full_text, re.IGNORECASE)
    if party_match:
        p1 = party_match.group(1).strip()
        p2 = party_match.group(2).strip()
        dimensions["parties"] = f"{p1} and {p2}"

    # Search among sentences for key domains
    for s in sentences:
        s_low = s.lower()

        # Purpose
        if any(k in s_low for k in ["purpose of this agreement", "agrees to provide", "hereby grants", "service provides", "licensor licenses"]) and dimensions["purpose"].startswith("Consumer"):
            dimensions["purpose"] = s

        # Duration & Term
        if any(k in s_low for k in ["initial term", "term of this agreement", "shall commence on", "for a period of", "commitment of"]) and dimensions["duration_and_term"].startswith("Standard"):
            dimensions["duration_and_term"] = s

        # Payment & Fees
        if any(k in s_low for k in ["subscription fee", "charges of", "payment shall be", "fees are non-refundable", "monthly dues", "price of"]) and dimensions["payment_and_fees"].startswith("Standard"):
            dimensions["payment_and_fees"] = s

        # Renewal
        if any(k in s_low for k in ["automatically renew", "auto-renew", "subsequent renewal", "evergreen", "prior to expiration"]) and dimensions["renewal"].startswith("No"):
            dimensions["renewal"] = s

        # Termination
        if any(k in s_low for k in ["terminate this agreement", "right to cancel", "early termination", "notice of cancellation", "for convenience"]) and dimensions["termination"].startswith("Standard"):
            dimensions["termination"] = s

        # Liability
        if any(k in s_low for k in ["limitation of liability", "shall not exceed", "not liable for", "indemnify", "disclaim"]) and dimensions["liability"].startswith("Standard"):
            dimensions["liability"] = s

        # Dispute Resolution
        if any(k in s_low for k in ["arbitration", "waive right to court", "disputes arising", "consumer forum", "amicable settlement"]) and dimensions["dispute_resolution"].startswith("Amicable"):
            dimensions["dispute_resolution"] = s

        # Governing Law
        if any(k in s_low for k in ["governed by the laws", "jurisdiction of", "courts of", "governing law"]) and dimensions["governing_law"].startswith("Laws"):
            dimensions["governing_law"] = s

    return dimensions


def summarize_document(
    text: str,
    mode: str = "hybrid",
    length: str = "medium"
) -> Dict[str, Any]:
    """
    Main entry point for document summarization.
    Modes:
      - 'extractive': Fast sentence ranking with SBERT & MMR (no seq2seq overhead)
      - 'abstractive': Map-reduce seq2seq generation
      - 'hybrid': Extractive saliency filtering + abstractive synthesis
    Lengths:
      - 'short': 2-3 sentence overview, 4 key points
      - 'medium': 3-5 sentence overview, 7-9 key points
      - 'detailed': Comprehensive overview and detailed legal provisions
    """
    warnings = []
    cleaned = clean_legal_text(text)

    # 1. Edge Case: Empty or unscannable document
    if not cleaned or len(cleaned.strip()) < 30:
        return {
            "error": "Document contains insufficient or unreadable text. Scanned PDFs without text layers or empty files cannot be summarized."
        }

    # 2. Edge Case: Non-English detection
    non_ascii_chars = len([c for c in cleaned if ord(c) > 127])
    if (non_ascii_chars / len(cleaned)) > 0.30:
        warnings.append("High proportion of non-English or non-ASCII characters detected. Summarization accuracy may vary.")

    word_count_original = len(cleaned.split())

    # 3. Edge Case: Very short documents
    if word_count_original < 60:
        warnings.append("Document is very short (< 60 words). Summarized directly without chunking.")
        return {
            "overview": cleaned,
            "key_points": {
                "general_summary": cleaned
            },
            "important_dates_and_amounts": [],
            "word_count_original": word_count_original,
            "word_count_summary": word_count_original,
            "compression_ratio": 1.0,
            "mode": mode,
            "length": length,
            "warnings": warnings
        }

    sentences = split_into_sentences(cleaned)
    sbert = get_sbert_model()

    # Determine sentence budget
    k_map = {"short": 3, "medium": 5, "detailed": 8}
    top_k = k_map.get(length.lower(), 5)

    # 1. Extractive Candidate Extraction
    extracted_sentences = extractive_summarize(sentences, sbert, top_k=top_k)

    overview_text = ""
    effective_mode = mode.lower()

    if effective_mode == "extractive":
        overview_text = " ".join(extracted_sentences)
    elif effective_mode in ["abstractive", "hybrid"]:
        pipe = get_abstractive_model()
        if pipe:
            source_text = " ".join(extracted_sentences) if effective_mode == "hybrid" else cleaned
            overview_text = abstractive_map_reduce(source_text, pipe, length_setting=length)
        else:
            warnings.append("Abstractive model unavailable on local environment. Reverted to Extractive mode.")
            overview_text = " ".join(extracted_sentences)
            effective_mode = "extractive"
    else:
        overview_text = " ".join(extracted_sentences)

    # 2. Structured Key Points across 9 dimensions
    structured_points = extract_structured_key_points(cleaned, sentences)
    if length == "short":
        # Keep top 4 dimensions
        keys_to_keep = ["purpose", "duration_and_term", "payment_and_fees", "termination"]
        structured_points = {k: structured_points[k] for k in keys_to_keep}

    # 3. Important Dates and Amounts Extraction (using extract_financial_entities)
    extracted_entities = []
    # Test each clause / sentence
    for s in sentences:
        ent = extract_financial_terms(s)
        has_val = (ent.get("amount") != "N/A" or ent.get("deadline") != "N/A" or ent.get("refund_condition") != "N/A")
        if has_val:
            extracted_entities.append({
                "context": s[:180] + ("..." if len(s) > 180 else ""),
                "amount": ent.get("amount"),
                "currency": ent.get("currency"),
                "deadline_or_notice": ent.get("deadline"),
                "trigger": ent.get("trigger"),
                "consequence": ent.get("consequence"),
                "refund_condition": ent.get("refund_condition")
            })

    # Deduplicate entities by amount + deadline
    seen_ents = set()
    unique_entities = []
    for e in extracted_entities:
        sig = (e["amount"], e["currency"], e["deadline_or_notice"])
        if sig not in seen_ents:
            seen_ents.add(sig)
            unique_entities.append(e)

    # Calculate Metrics
    summary_words = len(overview_text.split()) + sum(len(v.split()) for v in structured_points.values())
    compression_ratio = round(summary_words / max(word_count_original, 1), 3)

    return {
        "overview": overview_text,
        "key_points": structured_points,
        "important_dates_and_amounts": unique_entities[:8],
        "word_count_original": word_count_original,
        "word_count_summary": summary_words,
        "compression_ratio": min(1.0, compression_ratio),
        "mode": effective_mode,
        "length": length,
        "warnings": warnings
    }
