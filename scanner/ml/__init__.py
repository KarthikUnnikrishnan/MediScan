"""
scanner/ml — Precision Offline Medicine Scanning Package.

Architecture:
- vision.py:            Custom YOLOv8 Strip Detector & Fine-tuned TrOCR model (100% offline)
- salt_parser.py:       Active Ingredient & Chemical Salt Formulation Parser (100% offline)
- extractor.py:         Context-Aware Normalizer & Structured OCR Extractor (100% offline)
- medicine_index.py:    Dynamic 503k+ Medicine Search Index with Fast Cache (100% offline)
- medicine_matcher.py:  Multi-Tier Matcher with Hard Composition Validation (100% offline)
- alternative_matcher.py: Bioequivalent Cheaper Alternatives Search Engine (100% offline)
- drug_safety.py:       Active Chemical Side Effects & Drug Interaction Engine (100% offline)

100% Offline Deep Learning Pipeline:
  1. YOLOv8 object detector locates blister strip / packaging boundaries
  2. Fine-tuned TrOCR model reads medicine branding, dosage, and composition
  3. Context-aware NLP extracts brand, active salts, strengths, and dosage forms
  4. Multi-tier SQLite search validates candidate identity against 503k+ medicines
  5. Bioequivalent cheaper alternatives, MedDRA side effects, and drug interactions
     are retrieved purely from local SQLite databases.
"""

import logging
import os
import re
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image, ImageEnhance

from .vision import load_models, crop_strip, run_ocr
from .extractor import (
    extract_structured_ocr,
    normalize_ocr_text,
    extract_dosage_form,
    build_descriptive_medicine_name,
)
from .medicine_index import get_medicine_index
from .medicine_matcher import match_medicine, CONFIDENCE_THRESHOLD
from .alternative_matcher import find_top_cheaper_alternatives
from .drug_safety import get_drug_safety_info, get_cross_drug_interactions
from .salt_parser import parse_salt
from .price_verifier import KNOWN_BRAND_MRPS

logger = logging.getLogger(__name__)


# ── TrOCR Quality Heuristics ──────────────────────────────────────────────────

def _check_trocr_credibility(trocr_text: str) -> Tuple[bool, str]:
    """
    Evaluate whether TrOCR output is credible enough to attempt candidate matching.
    Uses purely algorithmic and structural heuristics without external network calls.
    """
    if not trocr_text or not trocr_text.strip():
        return False, "TrOCR returned empty text"

    cleaned = trocr_text.strip()

    # Heuristic 1: Text length check
    if len(cleaned) < 3:
        return False, f"TrOCR output too short ({len(cleaned)} chars)"

    # Heuristic 2: Insufficient alphanumeric tokens
    alphanumeric = re.sub(r'[^a-zA-Z0-9]', '', cleaned)
    if len(alphanumeric) < 3:
        return False, "TrOCR output has insufficient alphanumeric characters"

    # Heuristic 3: Excessive token repetition
    words = cleaned.split()
    if len(words) >= 4 and len(set(words)) == 1:
        return False, "TrOCR output exhibits repetitive character hallucination"

    return True, "TrOCR output passed credibility heuristics"


def is_prescription_text(text: str) -> bool:
    """
    Check if the text contains multiple clear markers of a doctor's prescription.
    """
    if not text:
        return False
    rx_patterns = [
        r'\b(?:m\.?d\.?|mbbs|dr\.|doctor|clinic|hospital|physician)\b',
        r'\b(?:rx|r/x|advice|treatment)\b',
        r'\b(?:sig:?|tabs?\s*no\.?|take\s+\d+|every\s*day|refill)\b',
        r'\b(?:syp|syrup)\s+[a-z0-9\-]+',
        r'\b(?:tds|sos|q\d+h|od|bd|qid|hs)\b',
        r'\b(?:dea\s*no|license\s*no|reg\.?\s*no)\b',
        r'\b(?:clinical\s*description|urti|lrti|diagnosis|patient\s*name)\b',
    ]
    matches = sum(1 for p in rx_patterns if re.search(p, text, re.I))
    return matches >= 2


# ── Medicine package scan (100% Offline Model Driven) ─────────────────────────

def scan_medicine_package(
    image_pil,
    cropped_pil: Optional[Any] = None,
    yolo_hit: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Execute 100% offline medicine package scanning:
    1. Reads printed text lines via PrintedRecogniser (Model A, fp16 on GPU).
    2. Matches recognized lines against the local database with match_printed_lines.
    3. If verified (top candidate confident + corroborated with salt/strength):
       retrieves bioequivalent cheaper alternatives and clinical safety/interactions.
    4. If unverified: returns unverified status with top 3 candidates and rejection reasons.
    """
    try:
        from .recognisers import get_printed_recogniser
        from .medicine_matcher import match_printed_lines
        from .salt_parser import parse_salt
        from .alternative_matcher import find_top_cheaper_alternatives
        from .drug_safety import get_drug_safety_info
        from django.conf import settings

        use_crop = getattr(settings, 'USE_STRIP_CROP', False)
        target_img = image_pil
        if use_crop:
            try:
                from .vision import crop_strip
                cropped, hit = crop_strip(image_pil)
                if hit and cropped:
                    target_img = cropped
            except Exception as e:
                logger.debug("Strip crop skipped: %s", e)

        rec_a = get_printed_recogniser()
        lines = rec_a.read_lines(target_img, beams=1)
        match_res = match_printed_lines(lines)

        status = match_res.get("status", "unverified")
        match_type = match_res.get("match_type", "brand")
        candidates = match_res.get("candidates", [])
        raw_ocr = match_res.get("raw_ocr", " ".join(lines))
        confidence_breakdown = match_res.get("confidence_breakdown", {
            "brand": 0.0, "ingredients": 0.0, "strength": 0.0,
            "manufacturer": 0.0, "dosage_form": 0.0, "overall": 0.0
        })

        if status != "verified" or not match_res.get("medicine"):
            return {
                "success": False,
                "status": "unverified",
                "match_type": match_type,
                "mode": "medicine_package",
                "message": (
                    "Unable to confidently verify this medicine. "
                    "Returning unverified is safer than returning a wrong result."
                ),
                "rejection_reasons": (
                    [match_res.get("message")] if match_res.get("message")
                    else (["Top candidate was not confident and corroborated."] if candidates else ["No readable brand text recognized."])
                ),
                "raw_ocr": raw_ocr,
                "medicine": None,
                "candidates": candidates,
                "medicines": candidates,
                "alternatives": [],
                "side_effects": [],
                "interactions": [],
                "confidence_breakdown": confidence_breakdown,
            }

        verified_med = match_res["medicine"]
        mfr = (verified_med.get("manufacturer") or "").strip() or "Registered Manufacturer"
        verified_med["manufacturer"] = mfr

        # Bioequivalent Alternatives Search (Local SQLite)
        if not verified_med.get("parsed_salt") and verified_med.get("salt"):
            verified_med["parsed_salt"] = parse_salt(verified_med["salt"])
        alternatives = find_top_cheaper_alternatives(verified_med, limit=5)

        max_savings = None
        max_savings_percent = None
        for alt in alternatives:
            s_amt = alt.get("saving_amount")
            if s_amt is not None:
                if max_savings is None or s_amt > max_savings:
                    max_savings = s_amt
                    max_savings_percent = alt.get("saving_percent")

        # Clinical Safety Lookup (Local SQLite)
        ingredients_list = verified_med.get("parsed_salt") or parse_salt(verified_med.get("salt", ""))
        safety_info = get_drug_safety_info(ingredients_list)

        schedule = "Schedule H"
        for ing in ingredients_list:
            cname = ing.get("canonical_name", "").lower()
            if any(k in cname for k in ["paracetamol", "calcium", "vitamin", "cholecalciferol"]):
                schedule = "General / OTC"

        display_ingredients = [
            {
                "name": ing.get("ingredient", "").title(),
                "canonical_name": ing.get("canonical_name", ""),
                "strength": ing.get("strength"),
                "unit": ing.get("unit", "mg"),
            }
            for ing in ingredients_list
        ]

        brand_display = verified_med.get("brand") or verified_med.get("name", "").split()[0]
        descriptive_name = verified_med.get("name", "Medicine")
        verified_med["detected_name"] = descriptive_name
        verified_med["brand"] = brand_display

        return {
            "success": True,
            "status": "verified",
            "match_type": match_type,
            "mode": "medicine_package",
            "detected_name": descriptive_name,
            "medicine": {
                "id": verified_med.get("id"),
                "name": verified_med.get("name"),
                "detected_name": descriptive_name,
                "brand": brand_display,
                "manufacturer": verified_med.get("manufacturer"),
                "dosage_form": verified_med.get("dosage_form", "Tablet"),
                "price": verified_med.get("price"),
                "salt": verified_med.get("salt"),
                "schedule": schedule,
                "confidence": confidence_breakdown.get("overall", 0.0),
                "confident": verified_med.get("confident", True),
                "corroborated": verified_med.get("corroborated", True),
            },
            "candidates": candidates,
            "medicines": [verified_med] + [c for c in candidates if c.get("id") != verified_med.get("id")][:2],
            "ingredients": display_ingredients,
            "alternatives": alternatives,
            "max_savings": max_savings,
            "max_savings_percent": max_savings_percent,
            "side_effects": safety_info.get("side_effects", []),
            "interactions": safety_info.get("interactions", []),
            "raw_ocr": raw_ocr,
            "confidence_breakdown": confidence_breakdown,
            "verification": {
                "matched_by": "Deep Learning Vision (PrintedRecogniser Model A) & Local SQLite Validation",
                "confidence": confidence_breakdown.get("overall", 0.0),
                "breakdown": confidence_breakdown,
            }
        }

    except Exception as exc:
        logger.error("scan_medicine_package failed: %s", exc, exc_info=True)
        return {
            "success": False,
            "status": "unverified",
            "error": str(exc),
            "raw_ocr": "",
            "candidates": [],
            "medicines": [],
        }

        # Clinical Safety Lookup (Local SQLite)
        ingredients_list = (
            verified_med.get("parsed_salt") or
            (best_extracted.get("ingredients") if best_extracted else []) or []
        )
        safety_info = get_drug_safety_info(ingredients_list)

        schedule = "Schedule H"
        for ing in ingredients_list:
            cname = ing.get("canonical_name", "").lower()
            if any(k in cname for k in ["paracetamol", "calcium", "vitamin", "cholecalciferol"]):
                schedule = "General / OTC"

        display_ingredients = [
            {
                "name": ing.get("ingredient", "").title(),
                "canonical_name": ing.get("canonical_name", ""),
                "strength": ing.get("strength"),
                "unit": ing.get("unit", "mg"),
            }
            for ing in ingredients_list
        ]

        brand_display = (
            (best_extracted.get("brand") if best_extracted else None) or
            verified_med.get("name", "").split()[0]
        )

        generic_label = best_extracted.get("generic") if best_extracted else None
        descriptive_name = build_descriptive_medicine_name(
            verified_med=verified_med,
            extracted=best_extracted,
            generic_text=generic_label,
        )

        verified_med["detected_name"] = descriptive_name
        verified_med["brand"] = brand_display

        return {
            "success": True,
            "status": "verified",
            "mode": "medicine_package",
            "detected_name": descriptive_name,
            "medicine": {
                "id": verified_med.get("id"),
                "name": verified_med.get("name"),
                "detected_name": descriptive_name,
                "brand": brand_display,
                "manufacturer": verified_med.get("manufacturer") or "Registered Manufacturer",
                "dosage_form": verified_med.get("dosage_form") or (best_extracted.get("dosage_form") if best_extracted else None) or "Tablet",
                "price": verified_med.get("price"),
                "salt": verified_med.get("salt"),
                "schedule": schedule,
                "confidence": confidence.get("overall", 0.0),
                "price_verification": verified_med.get("price_verification"),
            },
            "medicines": [verified_med],
            "ingredients": display_ingredients,
            "alternatives": alternatives,
            "max_savings": max_savings,
            "max_savings_percent": max_savings_percent,
            "side_effects": safety_info.get("side_effects", []),
            "interactions": safety_info.get("interactions", []),
            "raw_ocr": best_ocr_text,
            "raw_ocr_trocr": ocr_crop or ocr_full or "",
            "recovery_used": False,
            "recovery_reason": None,
            "disagreement_detected": False,
            "normalized_ocr": best_extracted.get("normalized_ocr", "") if best_extracted else "",
            "confidence_breakdown": confidence,
            "verification": {
                "matched_by": "Deep Learning Vision (YOLOv8 + TrOCR) & Local Database Validation",
                "confidence": confidence.get("overall", 0.0),
                "breakdown": confidence,
            }
        }

    except Exception as exc:
        logger.error("scan_medicine_package failed: %s", exc, exc_info=True)
        return {
            "success": False,
            "status": "unverified",
            "error": str(exc),
            "raw_ocr": "",
            "medicines": [],
        }


# ── Prescription scan helpers ──────────────────────────────────────────────────

_RX_METADATA_IGNORE = re.compile(
    r'\b(?:m\.?d\.?|mbbs|dr\.|doctor|clinic|hospital|chc|phc|healthcare|pharmacy|physician|'
    r'state\s*license|dea\s*no|license\s*no|reg\.?\s*no|phone|tel|fax|email|street|road|avenue|'
    r'lane|suite|pincode|zip\s*code|nemmara|thrissur|jipmer|maryland|libertyville)\b'
    r'|^\s*(?:name|patient|age|sex|gender|date|address|uhid|opd|ipd|weight|wt)\b'
    r'|^\s*(?:clinical\s*description|advice|adv:|rx:|treatment|diagnosis|symptoms?)\b'
    r'|^\s*(?:c/o|imp:|o/e|urti|lrti|fever|cough|cold|rr\s*[-:]|rs\s*[-:]|bp\s*[-:]|pr\s*[-:]|spo2|rbs)\b'
    r'|^\s*(?:refill|label|generic\s*if\s*available|signature|directions?|sig\b|every\s*day|take\s|with\s*water)\b'
    r'|^\s*(?:\d+(?:\.\d+)?\s*(?:mg|ml|mcg|g|iu|%)(?!\w)|tabs?|caps?|nos?\.?\s*\d+)\s*$'
    r'|^\s*(?:tabs?|caps?|tablets?|capsules?|syrup|syp|suspension|inj|injections?|drops?|solution|cream|gel|ointment)\s*$'
    r'|^\s*[:\-\.]+\s*'
    r'|^\s*(?:ph|tel|mob|mobile|cell)\s*[:\.]',
    re.IGNORECASE
)


def extract_prescription_candidates(ocr_text: str) -> List[str]:
    """
    Extract medicine query strings from prescription OCR text.
    Filters out doctor/clinic headers, patient demographics, and dosage directions.
    """
    candidates = []
    seen = set()

    def _add(cand: Any):
        if not cand or not isinstance(cand, str):
            return
        c = cand.strip()
        c = re.sub(r'[^\x00-\x7F]+', ' ', c).strip()
        c = re.sub(r'\((\d+(?:\.\d+)?)\s*/\s*\d+(?:\.\d+)?\)', r'\1mg', c)
        c = re.sub(r'\b(\d+(?:\.\d+)?)\s*/\s*\d+(?:\.\d+)?\b', r'\1mg', c)
        c = re.sub(r'\b(?:tabs?|caps?|no\.?|nos\.?)\s*\d+.*$', '', c, flags=re.I).strip()
        c = re.sub(r'(?<!/)\b\d+(?:\.\d+)?\s*ml\b.*$', '', c, flags=re.I).strip()
        c = re.sub(r'\b(?:sig:?|od|bd|tds|qid|hs|sos|once|twice|daily|every\s*day|q\d+h).*$', '', c, flags=re.I).strip()
        c = re.sub(r'\bx\s*\d+\s*[dhmw].*$', '', c, flags=re.I).strip()
        c = re.sub(r'[-–—]\s*\d+.*$', '', c).strip()
        c = re.sub(r'^(?:rx|r/x|adv(?:ice)?[:\.]?|->|[*•]|\(?\d+\)?[\.\s]*)\s*', '', c, flags=re.I).strip()
        c = re.sub(r'\s*\(\s*(?:iv|im|sc|po|stat)\s*\)', '', c, flags=re.I).strip()
        c = re.sub(r'\b(?:iv|im|sc|po|stat)\b.*$', '', c, flags=re.I).strip()
        c = re.sub(r'^[:\-\.\s]+', '', c).strip()
        c = re.sub(r'\s+', ' ', c).strip()
        if len(c) >= 3 and not _RX_METADATA_IGNORE.search(c):
            if re.search(r'[a-zA-Z]{2,}', c):
                key = c.lower()
                if key not in seen:
                    seen.add(key)
                    candidates.append(c)

    if not ocr_text:
        return candidates

    rx_match = re.search(r'\b(?:advice|treatment|rx|r/x)\b[:\s]*(.*)', ocr_text, flags=re.I | re.DOTALL)
    rx_body = rx_match.group(1) if rx_match else ""

    raw_lines = [l.strip() for l in re.split(r'[\r\n;]+', ocr_text) if l.strip()]
    rx_lines = [l.strip() for l in re.split(r'[\r\n;]+', rx_body) if l.strip()] if rx_body else []

    for line in rx_lines + raw_lines:
        if not re.search(r'[a-zA-Z]', line):
            continue
        if _RX_METADATA_IGNORE.search(line):
            continue

        if re.search(r'\b(?:rx|r/x|advice|treatment)\b', line, re.I):
            sub = re.split(r'\b(?:rx|r/x|advice|treatment)\b[:\s]*', line, flags=re.I)[-1].strip()
            if sub:
                _add(sub)
                continue

        if re.search(r'\b(?:tab|cap|inj|syp|syr|susp|oint|drop|resp)\b', line, re.I) or re.search(r'\d+(?:\.\d+)?\s*(?:mg|ml|mcg|g|iu|%)\b', line, re.I):
            _add(line)
        elif len(line) >= 4 and not any(w in line.lower() for w in ['name', 'date', 'street', 'phone', 'address', 'dr.', 'mbbs', 'chc', 'phc', 'reg.']):
            _add(line)

    return candidates


# ── Prescription scan (100% Offline Model Driven) ─────────────────────────────

def scan_prescription(image_pil) -> Dict[str, Any]:
    """
    Execute prescription scan workflow using Model B stub (or model when installed).
    """
    try:
        from .recognisers import get_handwritten_recogniser
        rec_b = get_handwritten_recogniser()
        res_b = rec_b.read_prescription(image_pil)

        msg = res_b.get("message", "Handwritten model is not installed yet")
        return {
            "success": False,
            "status": "unverified",
            "mode": "prescription",
            "message": msg,
            "error": msg,
            "raw_ocr": "",
            "candidates": [],
            "medicines": [],
            "per_medicine_results": {},
        }
    except Exception as exc:
        logger.error("scan_prescription failed: %s", exc, exc_info=True)
        return {
            "success": False,
            "status": "unverified",
            "mode": "prescription",
            "error": str(exc),
            "message": str(exc),
            "candidates": [],
            "medicines": [],
            "per_medicine_results": {},
        }


# ── Main entry point ──────────────────────────────────────────────────────────

def scan_image(image_path: str, mode: str = "medicine_package") -> Dict[str, Any]:
    """
    Main entry point called by Django views.
    Executes 100% offline with separate models per mode.
    """
    try:
        from PIL import Image
        image_pil = Image.open(image_path)
        if mode == "prescription":
            return scan_prescription(image_pil)
        return scan_medicine_package(image_pil)
    except Exception as exc:
        logger.error("scan_image failed on %s: %s", image_path, exc, exc_info=True)
        return {
            "success": False,
            "status": "unverified",
            "error": str(exc),
            "candidates": [],
            "medicines": []
        }




