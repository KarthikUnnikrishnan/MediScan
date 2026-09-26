"""
scanner/ml/ai_verifier.py — Secondary AI verifier for ambiguous or close candidates.

CRITICAL RULE:
Do NOT make an LLM the primary matcher.
Use AI only as a secondary verifier when:
- Confidence is borderline
- OCR is ambiguous
- Top 2 database candidates have nearly identical confidence (e.g. 96% vs 95%)

Returns ONLY:
- "PASS"
- "REJECT"
- "NEEDS_REVIEW"
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


def secondary_ai_verification(
    extracted_ocr: Dict[str, Any],
    candidate_a: Dict[str, Any],
    candidate_b: Optional[Dict[str, Any]] = None
) -> str:
    """
    Evaluate candidate suitability when database matching is borderline or ambiguous.

    Returns
    -------
    str:
        "PASS" | "REJECT" | "NEEDS_REVIEW"
    """
    raw_ocr = (extracted_ocr.get("raw_ocr") or "").lower()
    norm_ocr = (extracted_ocr.get("normalized_ocr") or "").lower()
    detected_ingredients = extracted_ocr.get("ingredients") or []

    med_a = candidate_a.get("candidate", candidate_a)
    name_a = (med_a.get("name") or "").lower()
    salts_a = med_a.get("parsed_salt") or []

    # Check if candidate A's active composition is supported by the OCR text
    if salts_a:
        unsupported_count = 0
        for salt_item in salts_a:
            cname = salt_item.get("canonical_name", "").lower()
            # If canonical name is not in OCR at all
            if cname and cname not in norm_ocr and cname not in raw_ocr:
                # Check fuzzy match against OCR tokens
                ratio = fuzz.partial_ratio(cname, norm_ocr)
                if ratio < 75:
                    unsupported_count += 1

        if unsupported_count == len(salts_a):
            logger.info("AI Verifier: Candidate %s rejected (all salts missing in OCR)", name_a)
            return "REJECT"

    # If two candidates are extremely close, compare brand edit distance
    if candidate_b:
        med_b = candidate_b.get("candidate", candidate_b)
        name_b = (med_b.get("name") or "").lower()

        ratio_a = fuzz.token_sort_ratio(norm_ocr, name_a)
        ratio_b = fuzz.token_sort_ratio(norm_ocr, name_b)

        if abs(ratio_a - ratio_b) < 3:
            return "NEEDS_REVIEW"
        elif ratio_a > ratio_b:
            return "PASS"
        else:
            return "REJECT"

    return "PASS"
