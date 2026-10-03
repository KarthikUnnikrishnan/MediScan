"""
scanner/ml/ai_validator.py
Groq LLaMA 3.1 70B AI validation layer for MediScan.
Runs after the main pipeline to validate results before display.
"""

import os
import json
import logging
import re
import time
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

_groq_client    = None
_groq_available = False
_GROQ_MODEL     = "llama-3.1-70b-versatile"


def init_groq():
    global _groq_client, _groq_available, _GROQ_MODEL
    try:
        from groq import Groq
        from django.conf import settings
        api_key = getattr(settings, 'GROQ_API_KEY', '') \
                  or os.environ.get('GROQ_API_KEY', '')
        if not api_key:
            logger.warning("GROQ_API_KEY not set — AI validation disabled")
            return
        _groq_client    = Groq(api_key=api_key)
        _groq_available = True

        # Verify model availability; if requested model is decommissioned or unavailable, pick active model
        try:
            available_ids = [m.id for m in _groq_client.models.list().data]
            if _GROQ_MODEL not in available_ids:
                for cand in [
                    "openai/gpt-oss-120b",
                    "qwen/qwen3.8-27b",
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                    "openai/gpt-oss-20b",
                ]:
                    if cand in available_ids:
                        _GROQ_MODEL = cand
                        break
        except Exception as model_check_err:
            logger.debug("Groq model check skipped: %s", model_check_err)

        logger.info("Groq AI validator ready (%s)", _GROQ_MODEL)
    except ImportError:
        logger.warning("groq not installed — AI validation disabled")
    except Exception as exc:
        logger.warning("Groq init failed: %s", exc)


def is_available() -> bool:
    return _groq_available


def _call_groq(system: str, user: str, retries: int = 2) -> Optional[str]:
    if not _groq_available or not _groq_client:
        return None
    for attempt in range(1, retries + 1):
        try:
            resp = _groq_client.chat.completions.create(
                model    = _GROQ_MODEL,
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature     = 0.1,
                max_tokens      = 512,
                response_format = {"type": "json_object"},
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            err = str(exc)
            if "429" in err or "rate" in err.lower():
                wait = 10 * attempt
                logger.warning("Groq rate limited, waiting %ds...", wait)
                time.sleep(wait)
            else:
                logger.warning("Groq error (attempt %d): %s", attempt, exc)
                break
    return None


def _safe_json(text: Optional[str]) -> Optional[Dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


def validate_ocr_and_match(
    ocr_text: str,
    matched_name: str,
    matched_salt: str,
) -> Dict[str, Any]:
    default = {
        "ocr_corrected":     ocr_text,
        "match_valid":       True,
        "match_explanation": "Accepted without AI validation.",
        "confidence_label":  "Medium",
        "corrected_name":    None,
    }
    if not _groq_available:
        return default

    system = (
        "You are a clinical pharmacist AI validating medicine OCR results "
        "from medicine strip photos in India. "
        "Respond with valid JSON only. No extra text."
    )
    user = f"""A medicine strip photo was scanned with OCR.

RAW OCR TEXT:
\"\"\"{ocr_text}\"\"\"

DATABASE MATCH:
  Medicine : {matched_name}
  Salt     : {matched_salt}

Tasks:
1. Fix obvious OCR errors in the raw text
   (common: O/0, l/1, rn/m, missing spaces, wrong capitalisation)
2. Is this database match plausible for the OCR text?
3. If match is wrong, suggest the correct medicine name.

Reply ONLY with this JSON:
{{
  "ocr_corrected":     "<cleaned OCR text>",
  "match_valid":       <true or false>,
  "match_explanation": "<one sentence why match is correct or wrong>",
  "confidence_label":  "<High, Medium, or Low>",
  "corrected_name":    "<correct name or null>"
}}"""

    parsed = _safe_json(_call_groq(system, user))
    if not parsed:
        return default
    return {
        "ocr_corrected":     str(parsed.get("ocr_corrected", ocr_text)),
        "match_valid":       bool(parsed.get("match_valid", True)),
        "match_explanation": str(parsed.get("match_explanation", "")),
        "confidence_label":  str(parsed.get("confidence_label", "Medium")),
        "corrected_name":    parsed.get("corrected_name"),
    }


def validate_alternatives(
    detected_name: str,
    detected_salt: str,
    detected_price: Optional[float],
    alternatives: List[Dict],
) -> List[Dict]:
    if not _groq_available or not alternatives:
        return alternatives[:5]

    candidates = [
        {"index": i, "name": a.get("name",""),
         "salt": a.get("salt",""), "price": a.get("price",0)}
        for i, a in enumerate(alternatives[:8])
    ]

    system = (
        "You are a clinical pharmacist AI filtering generic medicine "
        "alternatives for bioequivalence in India. "
        "Respond with valid JSON only."
    )
    user = f"""Detected medicine:
  Name  : {detected_name}
  Salt  : {detected_salt}
  Price : Rs.{detected_price or 'unknown'}

Candidate cheaper alternatives:
{json.dumps(candidates, indent=2)}

Keep only alternatives with the SAME active ingredient.
Remove any that are a different drug or irrelevant.
Sort valid ones cheapest first. Maximum 5.

Reply ONLY with this JSON:
{{
  "valid_indices":  [<index numbers to keep, cheapest first>],
  "removed_reason": "<brief reason if any removed, else null>"
}}"""

    parsed = _safe_json(_call_groq(system, user))
    if not parsed or "valid_indices" not in parsed:
        return alternatives[:5]

    removed = parsed.get("removed_reason")
    if removed:
        logger.info("Groq removed alternatives: %s", removed)

    filtered = [
        alternatives[i]
        for i in parsed.get("valid_indices", [])
        if isinstance(i, int) and 0 <= i < len(alternatives)
    ]
    return filtered[:5] if filtered else alternatives[:5]


def run_ai_validation(scan_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point. Called from scan_image() after the pipeline.
    Enriches the result dict with AI validation data.
    Never breaks the app — always returns the result.
    """
    if not _groq_available:
        scan_result["ai_validation"] = {"available": False}
        return scan_result

    if not scan_result.get("success"):
        scan_result["ai_validation"] = {"available": False}
        return scan_result

    try:
        medicines    = scan_result.get("medicines", [])
        alternatives = scan_result.get("alternatives", [])
        ocr_text     = scan_result.get("ocr_text", "") or scan_result.get("raw_ocr", "") or scan_result.get("normalized_ocr", "")

        if not medicines:
            scan_result["ai_validation"] = {
                "available": True, "skipped": True
            }
            return scan_result

        top = medicines[0]

        # Validate OCR and match
        ocr_result = validate_ocr_and_match(
            ocr_text     = ocr_text,
            matched_name = top.get("name", ""),
            matched_salt = top.get("salt", ""),
        )

        # Filter alternatives
        if alternatives:
            scan_result["alternatives"] = validate_alternatives(
                detected_name  = top.get("name", ""),
                detected_salt  = top.get("salt", ""),
                detected_price = top.get("price"),
                alternatives   = alternatives,
            )

        scan_result["ai_validation"] = {
            "available":         True,
            "ocr_corrected":     ocr_result["ocr_corrected"],
            "match_valid":       ocr_result["match_valid"],
            "match_explanation": ocr_result["match_explanation"],
            "confidence_label":  ocr_result["confidence_label"],
            "corrected_name":    ocr_result["corrected_name"],
        }

        if not ocr_result["match_valid"] and ocr_result["corrected_name"]:
            scan_result["ai_suggested_correction"] = \
                ocr_result["corrected_name"]

        logger.info(
            "AI validation done — valid=%s confidence=%s",
            ocr_result["match_valid"],
            ocr_result["confidence_label"],
        )

    except Exception as exc:
        logger.error("run_ai_validation error: %s", exc, exc_info=True)
        scan_result["ai_validation"] = {
            "available": False, "error": str(exc)
        }

    return scan_result
