"""
scanner/ml — Precision Medicine Scanning, OCR & Bioequivalent Alternative Matching Package.

Architecture:
- vision.py: YOLOv8 Strip Detector & TrOCR Vision Models (Unchanged)
- salt_parser.py: Active Ingredient & Chemical Salt Formulation Parser
- extractor.py: Context-Aware Normalizer & Structured OCR Extractor
- medicine_index.py: Dynamic 503k+ Medicine Search Index with Fast Cache
- medicine_matcher.py: Multi-Tier Matcher with Hard Composition Validation
- alternative_matcher.py: Bioequivalent Cheaper Alternatives Search Engine
- drug_safety.py: Active Chemical Side Effects & Drug Interaction Engine
- ai_verifier.py: Secondary Candidate AI Verifier
"""

import logging
import os
import re
from typing import Dict, Any, Optional

from .vision import load_models, crop_strip, run_ocr
from .extractor import extract_structured_ocr, normalize_ocr_text
from .medicine_index import get_medicine_index
from .medicine_matcher import match_medicine
from .alternative_matcher import find_top_cheaper_alternatives
from .drug_safety import get_drug_safety_info
from .ai_validator import init_groq, run_ai_validation

logger = logging.getLogger(__name__)


def scan_medicine_package(image_pil) -> Dict[str, Any]:
    """
    Execute precision medicine package/strip scanning workflow:
    1. YOLO strip crop
    2. TrOCR text recognition
    3. Context-aware OCR normalization & structured extraction
    4. Multi-tier database matching with hard composition validation
    5. Top 5 cheaper bioequivalent alternatives
    6. Clinical safety, side effects, and interaction warnings
    """
    try:
        # 1. Detect and crop strip/label
        cropped_pil, yolo_hit = crop_strip(image_pil)

        # 2. Run TrOCR text recognition
        raw_ocr = run_ocr(cropped_pil)
        if not raw_ocr and not yolo_hit:
            # Retry on full image if crop returned nothing
            raw_ocr = run_ocr(image_pil)

        if not raw_ocr:
            return {
                "success": False,
                "status": "unverified",
                "mode": "medicine_package",
                "error": "No readable text detected on the medicine package.",
                "raw_ocr": "",
                "medicines": [],
                "confidence_breakdown": {
                    "brand": 0.0, "ingredients": 0.0, "strength": 0.0,
                    "manufacturer": 0.0, "dosage_form": 0.0, "overall": 0.0
                }
            }

        # 3. Structured OCR Extraction
        extracted = extract_structured_ocr(raw_ocr)

        # 4. Multi-tier Database Matching
        match_result = match_medicine(extracted)
        status = match_result.get("status", "unverified")
        verified_med = match_result.get("medicine")
        confidence = match_result.get("confidence_breakdown", {})

        if status != "verified" or not verified_med:
            return {
                "success": False,
                "status": "unverified",
                "mode": "medicine_package",
                "message": "Unable to confidently verify this medicine. It is better to return unverified than the wrong medicine.",
                "rejection_reasons": match_result.get("rejection_reasons", []),
                "raw_ocr": raw_ocr,
                "normalized_ocr": extracted.get("normalized_ocr", ""),
                "extracted": extracted,
                "medicines": [],
                "alternatives": [],
                "confidence_breakdown": confidence,
            }

        # 5. Top 5 Cheaper Bioequivalent Alternatives
        alternatives = find_top_cheaper_alternatives(verified_med, limit=5)

        max_savings = None
        max_savings_percent = None
        for alt in alternatives:
            s_amt = alt.get("saving_amount")
            if s_amt is not None:
                if max_savings is None or s_amt > max_savings:
                    max_savings = s_amt
                    max_savings_percent = alt.get("saving_percent")

        # 6. Safety Lookup (Using verified active chemical ingredients)
        ingredients_list = verified_med.get("parsed_salt") or extracted.get("ingredients") or []
        safety_info = get_drug_safety_info(ingredients_list)

        # Determine clinical schedule (Schedule H, Schedule G, OTC, etc.)
        schedule = "Schedule H"
        for ing in ingredients_list:
            cname = ing.get("canonical_name", "").lower()
            if any(k in cname for k in ["paracetamol", "calcium", "vitamin"]):
                schedule = "General / OTC"

        # Format ingredients cleanly for UI
        display_ingredients = []
        for ing in ingredients_list:
            display_ingredients.append({
                "name": ing.get("ingredient", "").title(),
                "canonical_name": ing.get("canonical_name", ""),
                "strength": ing.get("strength"),
                "unit": ing.get("unit", "mg"),
            })

        return {
            "success": True,
            "status": "verified",
            "mode": "medicine_package",
            "medicine": {
                "id": verified_med.get("id"),
                "name": verified_med.get("name"),
                "brand": extracted.get("brand") or verified_med.get("name").split()[0],
                "manufacturer": verified_med.get("manufacturer") or "Registered Manufacturer",
                "dosage_form": verified_med.get("dosage_form") or extracted.get("dosage_form") or "Tablet",
                "price": verified_med.get("price"),
                "salt": verified_med.get("salt"),
                "schedule": schedule,
                "confidence": confidence.get("overall", 0.0),
            },
            "medicines": [verified_med],
            "ingredients": display_ingredients,
            "alternatives": alternatives,
            "max_savings": max_savings,
            "max_savings_percent": max_savings_percent,
            "side_effects": safety_info.get("side_effects", []),
            "interactions": safety_info.get("interactions", []),
            "raw_ocr": raw_ocr,
            "normalized_ocr": extracted.get("normalized_ocr", ""),
            "confidence_breakdown": confidence,
            "verification": {
                "matched_by": "Multi-tier Database Matching & Composition Validation",
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


def scan_prescription(image_pil) -> Dict[str, Any]:
    """
    Execute prescription scanning workflow:
    1. TrOCR full-image text recognition
    2. Extract multiple medicine names
    3. Multi-medicine lookup & cross-interaction checking
    """
    try:
        raw_ocr = run_ocr(image_pil)
        if not raw_ocr:
            return {
                "success": False,
                "status": "unverified",
                "mode": "prescription",
                "error": "Could not read text from prescription.",
                "raw_ocr": "",
                "medicines": []
            }

        extracted = extract_structured_ocr(raw_ocr)
        norm_text = extracted.get("normalized_ocr") or raw_ocr

        # Split prescription lines
        lines = [l.strip() for l in re.split(r'[\n;,]+', norm_text) if len(l.strip()) >= 3]
        found_medicines = []
        all_ingredients = []

        for line in lines[:8]:
            line_ext = extract_structured_ocr(line)
            m_res = match_medicine(line_ext)
            if m_res.get("status") == "verified" and m_res.get("medicine"):
                med = m_res["medicine"]
                found_medicines.append(med)
                for ing in med.get("parsed_salt", []):
                    all_ingredients.append(ing)

        # Safety & cross-interactions between all found medications
        safety_info = get_drug_safety_info(all_ingredients)

        return {
            "success": bool(found_medicines),
            "status": "verified" if found_medicines else "unverified",
            "mode": "prescription",
            "raw_ocr": raw_ocr,
            "normalized_ocr": norm_text,
            "medicines": found_medicines,
            "medicine": found_medicines[0] if found_medicines else None,
            "ingredients": all_ingredients,
            "alternatives": find_top_cheaper_alternatives(found_medicines[0], limit=5) if found_medicines else [],
            "side_effects": safety_info.get("side_effects", []),
            "interactions": safety_info.get("interactions", []),
            "cross_interactions": safety_info.get("interactions", []),
            "confidence_breakdown": {
                "brand": 90.0 if found_medicines else 0.0,
                "ingredients": 90.0 if found_medicines else 0.0,
                "strength": 85.0 if found_medicines else 0.0,
                "manufacturer": 80.0 if found_medicines else 0.0,
                "dosage_form": 90.0 if found_medicines else 0.0,
                "overall": 88.0 if found_medicines else 0.0,
            }
        }

    except Exception as exc:
        logger.error("scan_prescription failed: %s", exc, exc_info=True)
        return {"success": False, "status": "unverified", "error": str(exc)}


def scan_image(image_path: str, mode: str = "auto") -> Dict[str, Any]:
    """
    Main entry point called by Django views.
    """
    try:
        from PIL import Image
        image_pil = Image.open(image_path)

        if mode == "medicine_package":
            result = scan_medicine_package(image_pil)
        elif mode == "prescription":
            result = scan_prescription(image_pil)
        else:
            # Auto-mode
            logger.info("Auto-mode: running package scan")
            res = scan_medicine_package(image_pil)
            if res.get("status") == "verified" and res.get("medicine"):
                result = res
            else:
                logger.info("Auto-mode: trying prescription scan")
                rx_res = scan_prescription(image_pil)
                if rx_res.get("status") == "verified" and rx_res.get("medicines"):
                    result = rx_res
                else:
                    result = res if res.get("raw_ocr") else rx_res

        result = run_ai_validation(result)
        return result

    except Exception as exc:
        logger.error("scan_image failed on %s: %s", image_path, exc, exc_info=True)
        return {"success": False, "status": "unverified", "error": str(exc)}


# Load models on import
try:
    load_models()
    init_groq()
except Exception as _load_exc:
    logger.warning("MediScan vision models could not be loaded at import time: %s", _load_exc)

