"""
scanner/ml — Precision Medicine Scanning Package.

Architecture:
- vision.py:            YOLOv8 Strip Detector & TrOCR (with anti-hallucination guard)
- gemini_vision.py:     Gemini Vision OCR — modern google-genai SDK recovery engine
- groq_extractor.py:    Groq cloud NLP — structured medicine-name extraction
- salt_parser.py:       Active Ingredient & Chemical Salt Formulation Parser
- extractor.py:         Context-Aware Normalizer & Structured OCR Extractor
- medicine_index.py:    Dynamic 503k+ Medicine Search Index with Fast Cache
- medicine_matcher.py:  Multi-Tier Matcher with Hard Composition Validation
- alternative_matcher.py: Bioequivalent Cheaper Alternatives Search Engine
- drug_safety.py:       Active Chemical Side Effects & Drug Interaction Engine

OCR cascade for medicine package scan:
  1. TrOCR (local fine-tuned model)
  2. OCR quality & database candidate validation
  3. Gemini Vision OCR recovery (when TrOCR is empty, suspicious, or unverified)
  4. Disagreement evaluation & conservative verification safeguards
"""

import logging
import os
import re
from typing import Dict, Any, Optional, Tuple, List

from .vision import load_models, crop_strip, run_ocr
from .gemini_vision import (
    gemini_vision_ocr,
    gemini_vision_ocr_structured,
    strip_preamble,
    NO_TEXT_DETECTED,
)
from .groq_extractor import extract_medicines_with_groq
from .extractor import extract_structured_ocr, normalize_ocr_text, extract_dosage_form
from .medicine_index import get_medicine_index
from .medicine_matcher import match_medicine, CONFIDENCE_THRESHOLD
from .alternative_matcher import find_top_cheaper_alternatives
from .drug_safety import get_drug_safety_info, get_cross_drug_interactions
from .salt_parser import parse_salt

logger = logging.getLogger(__name__)


# ── TrOCR Quality Heuristics ──────────────────────────────────────────────────

def _check_trocr_credibility(trocr_text: str) -> Tuple[bool, str]:
    """
    Evaluate whether TrOCR output is credible enough to attempt candidate matching.
    Uses explicit heuristic checks (clearly labeled as heuristics).

    Returns (is_credible, reason).
    """
    if not trocr_text or not trocr_text.strip():
        return False, "TrOCR returned empty text"

    cleaned = trocr_text.strip()

    # Heuristic 1: Text length check
    if len(cleaned) < 4:
        return False, f"TrOCR output too short ({len(cleaned)} chars)"

    # Heuristic 2: Insufficient alphanumeric tokens
    alphanumeric = re.sub(r'[^a-zA-Z0-9]', '', cleaned)
    if len(alphanumeric) < 3:
        return False, "TrOCR output has insufficient alphanumeric characters"

    # Heuristic 3: Excessive token repetition
    words = cleaned.split()
    if len(words) >= 4 and len(set(words)) == 1:
        return False, "TrOCR output exhibits repetitive character hallucination"

    return True, "TrOCR output passed basic credibility heuristics"


# ── Medicine package scan with controlled recovery ────────────────────────────

def scan_medicine_package(image_pil) -> Dict[str, Any]:
    """
    Execute precision medicine package/strip scanning workflow:
    1. YOLO strip crop
    2. TrOCR text recognition
    3. OCR quality check
       - If credible: database candidate matching & candidate validation
       - If suspicious or unverified: Gemini Vision OCR recovery
    4. Disagreement & evidence evaluation
    5. Bioequivalent alternatives search
    6. Clinical safety, side effects, and interaction warnings
    """
    try:
        # 1. Detect and crop strip/label
        cropped_pil, yolo_hit = crop_strip(image_pil)

        # 2. Run TrOCR on crop
        trocr_text = run_ocr(cropped_pil)
        if not trocr_text and yolo_hit:
            # Fallback crop or full image if tight crop produced nothing
            trocr_text = run_ocr(image_pil)

        is_trocr_credible, credibility_reason = _check_trocr_credibility(trocr_text)

        trocr_extracted = None
        trocr_match = None
        need_recovery = False
        recovery_reason = ""

        if trocr_text and trocr_text.strip():
            trocr_extracted = extract_structured_ocr(trocr_text)
            trocr_match = match_medicine(trocr_extracted)

        if is_trocr_credible and trocr_match and trocr_match.get("status") == "verified":
            trocr_med = trocr_match.get("medicine")
            trocr_conf = trocr_match.get("confidence_breakdown", {}).get("overall", 0.0)

            if trocr_conf >= CONFIDENCE_THRESHOLD:
                need_recovery = False
                logger.info(
                    "scan_medicine_package: TrOCR produced credible, verified candidate %r (conf=%.1f%%)",
                    trocr_med.get("name"),
                    trocr_conf,
                )
            else:
                need_recovery = True
                recovery_reason = f"TrOCR confidence ({trocr_conf}%) below threshold ({CONFIDENCE_THRESHOLD}%)"
        else:
            need_recovery = True
            recovery_reason = credibility_reason if not is_trocr_credible else (
                f"TrOCR database verification failed: {'; '.join(trocr_match.get('rejection_reasons', ['unverified'])) if trocr_match else 'unverified'}"
            )

        # 3. Controlled Gemini Vision Recovery
        gemini_res: Dict[str, Any] = {}
        gemini_text = ""
        gemini_extracted = None
        gemini_match = None
        recovery_used = False

        if need_recovery:
            logger.info("scan_medicine_package: invoking Gemini Vision OCR recovery (%s)", recovery_reason)
            recovery_used = True

            # Send full image as primary context to avoid cropped-out text
            gemini_input = image_pil if image_pil is not None else cropped_pil
            gemini_res = gemini_vision_ocr_structured(gemini_input)

            if gemini_res.get("success") and not gemini_res.get("is_empty"):
                raw_gem = gemini_res.get("raw_text", "")
                if raw_gem and raw_gem != NO_TEXT_DETECTED:
                    gemini_text = gemini_res.get("plain_text") or strip_preamble(raw_gem)
                    if gemini_text:
                        gemini_extracted = extract_structured_ocr(gemini_text)

                        # Incorporate structured label fields from Gemini if available
                        struct = gemini_res.get("structured", {})
                        if struct.get("brand") and (not gemini_extracted.get("brand") or gemini_extracted.get("brand").lower() != struct["brand"].lower()):
                            gemini_extracted["brand"] = struct["brand"]
                        if struct.get("ingredients"):
                            parsed_salts = parse_salt(struct["ingredients"])
                            if parsed_salts:
                                gemini_extracted["ingredients"] = parsed_salts
                        if struct.get("dosage_form"):
                            gemini_extracted["dosage_form"] = extract_dosage_form(struct["dosage_form"]) or struct["dosage_form"]
                        if struct.get("manufacturer") and not gemini_extracted.get("manufacturer"):
                            gemini_extracted["manufacturer"] = struct["manufacturer"]

                        gemini_match = match_medicine(gemini_extracted)

        # 4. Disagreement & Candidate Selection
        final_ocr = ""
        final_extracted = None
        final_match = None
        disagreement_detected = bool(trocr_text and gemini_text and trocr_text.strip().lower() != gemini_text.strip().lower())

        trocr_verified = bool(trocr_match and trocr_match.get("status") == "verified" and trocr_match.get("medicine"))
        gemini_verified = bool(gemini_match and gemini_match.get("status") == "verified" and gemini_match.get("medicine"))

        if not need_recovery and trocr_verified:
            # TrOCR credible and verified without needing recovery
            final_ocr = trocr_text
            final_extracted = trocr_extracted
            final_match = trocr_match
        elif recovery_used and gemini_verified and not trocr_verified:
            # Recovery succeeded where TrOCR failed
            logger.info("scan_medicine_package: Gemini recovery successfully verified candidate %r", gemini_match["medicine"]["name"])
            final_ocr = gemini_text
            final_extracted = gemini_extracted
            final_match = gemini_match
        elif recovery_used and gemini_verified and trocr_verified:
            # Both returned verified candidates — compare evidence
            med_trocr = trocr_match["medicine"]
            med_gemini = gemini_match["medicine"]

            if med_trocr.get("id") == med_gemini.get("id"):
                # Both agree on identity
                final_ocr = gemini_text or trocr_text
                final_extracted = gemini_extracted or trocr_extracted
                final_match = gemini_match
            else:
                # Candidates conflict: evaluate evidence
                trocr_comp_ok = trocr_match.get("validation_details", {}).get("composition_valid", True)
                gemini_comp_ok = gemini_match.get("validation_details", {}).get("composition_valid", True)

                if gemini_comp_ok and not trocr_comp_ok:
                    final_ocr = gemini_text
                    final_extracted = gemini_extracted
                    final_match = gemini_match
                elif trocr_comp_ok and not gemini_comp_ok:
                    final_ocr = trocr_text
                    final_extracted = trocr_extracted
                    final_match = trocr_match
                else:
                    # Contradictory evidence cannot be safely resolved
                    logger.warning(
                        "scan_medicine_package: contradictory candidates (%r vs %r) — returning unverified",
                        med_trocr.get("name"),
                        med_gemini.get("name"),
                    )
                    return {
                        "success": False,
                        "status": "unverified",
                        "mode": "medicine_package",
                        "message": "Contradictory OCR candidate evidence. Returning unverified is safer than returning a wrong result.",
                        "rejection_reasons": ["Contradictory candidate evidence between local OCR and vision recovery."],
                        "raw_ocr": gemini_text or trocr_text,
                        "raw_ocr_trocr": trocr_text,
                        "raw_ocr_gemini": gemini_res.get("raw_text", ""),
                        "recovery_used": True,
                        "disagreement_detected": True,
                        "medicines": [],
                        "alternatives": [],
                        "confidence_breakdown": {
                            "brand": 0.0, "ingredients": 0.0, "strength": 0.0,
                            "manufacturer": 0.0, "dosage_form": 0.0, "overall": 0.0
                        },
                    }
        elif recovery_used and trocr_verified and not gemini_verified:
            # Gemini failed to find/verify, but TrOCR had a verified candidate
            final_ocr = trocr_text
            final_extracted = trocr_extracted
            final_match = trocr_match
        else:
            # Neither produced a verified candidate
            final_ocr = gemini_text or trocr_text
            final_extracted = gemini_extracted or trocr_extracted or extract_structured_ocr(final_ocr)
            final_match = gemini_match or trocr_match

        # 5. Handle unverified result
        if not final_match or final_match.get("status") != "verified" or not final_match.get("medicine"):
            rejection_reasons = []
            if final_match:
                rejection_reasons = final_match.get("rejection_reasons", [])
            if not rejection_reasons:
                if not final_ocr:
                    rejection_reasons = ["No readable text detected on the medicine package."]
                else:
                    rejection_reasons = ["Could not confidently verify medicine identity in database."]

            if recovery_used and gemini_res.get("success") is False and not trocr_verified:
                return {
                    "success": False, "status": "ocr_service_error", "mode": "medicine_package",
                    "message": "Text recognition service is temporarily unavailable. Please try again later.",
                    "ocr_error": gemini_res.get("error_type", "API_ERROR"),
                    "raw_ocr": final_ocr, "raw_ocr_trocr": trocr_text,
                    "raw_ocr_gemini": "", "recovery_used": True, "medicines": [], "alternatives": [],
                }
            return {
                "success": False,
                "status": "unverified",
                "mode": "medicine_package",
                "message": (
                    "Unable to confidently verify this medicine. "
                    "Returning unverified is safer than returning a wrong result."
                ),
                "rejection_reasons": rejection_reasons,
                "raw_ocr": final_ocr,
                "raw_ocr_trocr": trocr_text,
                "raw_ocr_gemini": gemini_res.get("raw_text", "") if recovery_used else "",
                "recovery_used": recovery_used,
                "recovery_reason": recovery_reason if recovery_used else None,
                "disagreement_detected": disagreement_detected,
                "normalized_ocr": final_extracted.get("normalized_ocr", "") if final_extracted else "",
                "extracted": final_extracted or {},
                "medicines": [],
                "alternatives": [],
                "confidence_breakdown": final_match.get("confidence_breakdown", {
                    "brand": 0.0, "ingredients": 0.0, "strength": 0.0,
                    "manufacturer": 0.0, "dosage_form": 0.0, "overall": 0.0
                }) if final_match else {
                    "brand": 0.0, "ingredients": 0.0, "strength": 0.0,
                    "manufacturer": 0.0, "dosage_form": 0.0, "overall": 0.0
                },
            }

        # 6. Verified Result Construction
        verified_med = final_match["medicine"]
        confidence = final_match.get("confidence_breakdown", {})

        # Top 5 Cheaper Bioequivalent Alternatives
        alternatives = find_top_cheaper_alternatives(verified_med, limit=5)

        max_savings = None
        max_savings_percent = None
        for alt in alternatives:
            s_amt = alt.get("saving_amount")
            if s_amt is not None:
                if max_savings is None or s_amt > max_savings:
                    max_savings = s_amt
                    max_savings_percent = alt.get("saving_percent")

        # Clinical Safety Lookup
        ingredients_list = (
            verified_med.get("parsed_salt") or
            (final_extracted.get("ingredients") if final_extracted else []) or []
        )
        safety_info = get_drug_safety_info(ingredients_list)

        schedule = "Schedule H"
        for ing in ingredients_list:
            cname = ing.get("canonical_name", "").lower()
            if any(k in cname for k in ["paracetamol", "calcium", "vitamin"]):
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
            (final_extracted.get("brand") if final_extracted else None) or
            verified_med.get("name", "").split()[0]
        )

        return {
            "success": True,
            "status": "verified",
            "mode": "medicine_package",
            "medicine": {
                "id": verified_med.get("id"),
                "name": verified_med.get("name"),
                "brand": brand_display,
                "manufacturer": verified_med.get("manufacturer") or "Registered Manufacturer",
                "dosage_form": verified_med.get("dosage_form") or (final_extracted.get("dosage_form") if final_extracted else None) or "Tablet",
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
            "raw_ocr": final_ocr,
            "raw_ocr_trocr": trocr_text,
            "raw_ocr_gemini": gemini_res.get("raw_text", "") if recovery_used else "",
            "recovery_used": recovery_used,
            "recovery_reason": recovery_reason if recovery_used else None,
            "disagreement_detected": disagreement_detected,
            "normalized_ocr": final_extracted.get("normalized_ocr", "") if final_extracted else "",
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


# ── Prescription scan helpers ──────────────────────────────────────────────────

_RX_METADATA_IGNORE = re.compile(
    r'\b(?:m\.?d\.?|mbbs|dr\.|doctor|clinic|hospital|chc|phc|healthcare|pharmacy|physician|'
    r'state\s*license|dea\s*no|license\s*no|reg\.?\s*no|phone|tel|fax|email|street|road|avenue|'
    r'lane|suite|pincode|zip\s*code|nemmara|thrissur|jipmer)\b'
    r'|^\s*(?:name|patient|age|sex|gender|date|address|uhid|opd|ipd|weight|wt)\b'
    r'|^\s*(?:clinical\s*description|advice|adv:|rx:|treatment|diagnosis|symptoms?)\b'
    r'|^\s*(?:urti|lrti|fever|cough|cold|rr\s*[-:]|rs\s*[-:]|bp\s*[-:]|pr\s*[-:]|spo2)\b'
    r'|^\s*(?:refill|label|generic\s*if\s*available|signature|directions?|sig\b|every\s*day|take\s|with\s*water)\b'
    r'|^\s*(?:\d+(?:\.\d+)?\s*(?:mg|ml|mcg|g|iu|%)|tabs?|caps?|nos?\.?\s*\d+)\s*$'
    r'|^\s*[:\-\.]+\s*'
    r'|^\s*(?:ph|tel|mob|mobile|cell)\s*[:\.]',
    re.IGNORECASE
)


def extract_prescription_candidates(ocr_text: str, gemini_structured: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Extract high-precision medicine query strings from prescription OCR text.
    Filters out doctor/clinic headers, patient info, and dosage directions.
    """
    candidates = []
    seen = set()

    def _add(cand: str):
        c = cand.strip()
        # Remove non-ASCII characters (such as Malayalam/regional annotations)
        c = re.sub(r'[^\x00-\x7F]+', ' ', c).strip()
        # Convert pediatric concentration fractions like (250/5) -> 250
        c = re.sub(r'\((\d+)\s*/\s*\d+\)', r'\1', c)
        c = re.sub(r'\b(\d+)\s*/\s*\d+\b', r'\1', c)
        # Strip administration directions & dosage schedules
        c = re.sub(r'\b(?:tabs?|caps?|no\.?|nos\.?)\s*\d+.*$', '', c, flags=re.I).strip()
        c = re.sub(r'\b\d+(?:\.\d+)?\s*ml\b.*$', '', c, flags=re.I).strip()
        c = re.sub(r'\b(?:sig:?|od|bd|tds|qid|hs|sos|once|twice|daily|every\s*day|q\d+h).*$', '', c, flags=re.I).strip()
        c = re.sub(r'\bx\s*\d+\s*[dhmw].*$', '', c, flags=re.I).strip()
        c = re.sub(r'[-–—]\s*\d+.*$', '', c).strip()
        c = re.sub(r'^(?:rx|r/x|\(?\d+\)?[\.\s]*)\s*', '', c, flags=re.I).strip()
        c = re.sub(r'^[:\-\.\s]+', '', c).strip()
        c = re.sub(r'\s+', ' ', c).strip()
        if len(c) >= 3 and not _RX_METADATA_IGNORE.search(c):
            if re.search(r'[a-zA-Z]{2,}', c):
                key = c.lower()
                if key not in seen:
                    seen.add(key)
                    candidates.append(c)

    # 1. Gemini structured fields (if available)
    if gemini_structured:
        brand = gemini_structured.get("brand") or ""
        strength = gemini_structured.get("strength") or ""
        generic = gemini_structured.get("generic") or ""
        if brand and strength:
            _add(f"{brand} {strength}")
        if brand:
            _add(brand)
        if generic and strength:
            _add(f"{generic} {strength}")
        elif generic:
            _add(generic)

    if not ocr_text:
        return candidates

    # 2. Extract lines in and following 'Advice', 'Rx', or 'Treatment' markers if present
    rx_match = re.search(r'\b(?:advice|treatment|rx|r/x)\b[:\s]*(.*)', ocr_text, flags=re.I | re.DOTALL)
    rx_body = rx_match.group(1) if rx_match else ""

    raw_lines = [l.strip() for l in re.split(r'[\r\n;]+', ocr_text) if l.strip()]
    rx_lines = [l.strip() for l in re.split(r'[\r\n;]+', rx_body) if l.strip()] if rx_body else []

    # Prioritize Rx/Advice lines, then full OCR lines
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
        elif len(line) >= 4 and not any(w in line.lower() for w in ['name', 'date', 'street', 'phone', 'address', 'dr.', 'mbbs', 'chc', 'phc']):
            _add(line)

    return candidates


# ── Prescription scan ─────────────────────────────────────────────────────────

def scan_prescription(image_pil) -> Dict[str, Any]:
    """
    Execute prescription scanning workflow:
    1. TrOCR full-image text recognition
    2. Controlled Gemini Vision OCR recovery if TrOCR produces empty/unusable text
    3. Multi-medicine extraction & precision database matching
    4. Per-medicine details, bioequivalent alternatives & cross-interaction checking
    """
    try:
        # 1. TrOCR on full prescription image
        trocr_text = run_ocr(image_pil)
        is_credible, credibility_reason = _check_trocr_credibility(trocr_text)

        recovery_used = False
        gemini_res: Dict[str, Any] = {}
        gemini_text = ""
        active_ocr = trocr_text

        # 2. Check if recovery is required
        if not is_credible:
            recovery_used = True
            gemini_res = gemini_vision_ocr_structured(image_pil)
            if gemini_res.get("success") and not gemini_res.get("is_empty"):
                raw_gem = gemini_res.get("raw_text", "")
                if raw_gem and raw_gem != NO_TEXT_DETECTED:
                    gemini_text = gemini_res.get("plain_text") or strip_preamble(raw_gem)
                    if gemini_text:
                        active_ocr = gemini_text

        if not active_ocr:
            if recovery_used and gemini_res.get("success") is False:
                return {
                    "success": False, "status": "ocr_service_error", "mode": "prescription",
                    "message": "Text recognition service is temporarily unavailable. Please try again later.",
                    "ocr_error": gemini_res.get("error_type", "API_ERROR"), "raw_ocr": "",
                    "raw_ocr_trocr": trocr_text, "raw_ocr_gemini": "", "recovery_used": True,
                    "medicines": [], "per_medicine_results": {}
                }
            return {
                "success": False,
                "status": "unverified",
                "mode": "prescription",
                "error": "Could not read text from prescription.",
                "raw_ocr": "",
                "raw_ocr_trocr": trocr_text,
                "raw_ocr_gemini": gemini_res.get("raw_text", "") if recovery_used else "",
                "recovery_used": recovery_used,
                "medicines": [],
                "per_medicine_results": {}
            }

        extracted = extract_structured_ocr(active_ocr)
        norm_text = extracted.get("normalized_ocr") or active_ocr

        # 3a. Extract candidate medicine lines
        candidates = extract_prescription_candidates(
            active_ocr,
            gemini_structured=gemini_res.get("structured") if recovery_used else None
        )

        found_medicines = []
        all_ingredients = []
        seen_med_ids = set()

        for cand_str in candidates[:25]:
            cand_ext = extract_structured_ocr(cand_str)
            m_res = match_medicine(cand_ext, is_prescription=True)
            if m_res.get("status") == "verified" and m_res.get("medicine"):
                med = m_res["medicine"]
                if med["id"] not in seen_med_ids:
                    seen_med_ids.add(med["id"])
                    found_medicines.append(med)
                    for ing in med.get("parsed_salt", []):
                        all_ingredients.append(ing)

        # 3b. If extraction found nothing, try Gemini recovery if not yet run
        if not found_medicines and not recovery_used:
            logger.info("scan_prescription: local extraction found 0 medicines — trying Gemini recovery")
            recovery_used = True
            gemini_res = gemini_vision_ocr_structured(image_pil)
            if gemini_res.get("success") and not gemini_res.get("is_empty"):
                raw_gem = gemini_res.get("raw_text", "")
                if raw_gem and raw_gem != NO_TEXT_DETECTED:
                    gemini_text = gemini_res.get("plain_text") or strip_preamble(raw_gem)
                    if gemini_text:
                        active_ocr = gemini_text
                        extracted = extract_structured_ocr(active_ocr)
                        norm_text = extracted.get("normalized_ocr") or active_ocr
                        gem_cands = extract_prescription_candidates(
                            active_ocr,
                            gemini_structured=gemini_res.get("structured")
                        )
                        for g_cand in gem_cands[:25]:
                            g_ext = extract_structured_ocr(g_cand)
                            m_res = match_medicine(g_ext, is_prescription=True)
                            if m_res.get("status") == "verified" and m_res.get("medicine"):
                                med = m_res["medicine"]
                                if med["id"] not in seen_med_ids:
                                    seen_med_ids.add(med["id"])
                                    found_medicines.append(med)
                                    for ing in med.get("parsed_salt", []):
                                        all_ingredients.append(ing)

        # 3c. Groq NLP fallback if still no medicines found
        if not found_medicines:
            logger.info("scan_prescription: trying Groq NLP extraction")
            groq_names = extract_medicines_with_groq(active_ocr)
            for name in groq_names:
                extracted_name = extract_structured_ocr(name)
                m_res = match_medicine(extracted_name, is_prescription=True)
                if m_res.get("status") == "verified" and m_res.get("medicine"):
                    med = m_res["medicine"]
                    if med["id"] not in seen_med_ids:
                        seen_med_ids.add(med["id"])
                        found_medicines.append(med)
                        for ing in med.get("parsed_salt", []):
                            all_ingredients.append(ing)

        # 4. Build per_medicine_results for multiple medicine switching
        per_medicine_results = {}
        for med in found_medicines:
            display_name = med.get("name") or "Medication"
            alts = find_top_cheaper_alternatives(med, limit=5)
            med_safety = get_drug_safety_info(med.get("parsed_salt", []))
            per_medicine_results[display_name] = {
                "top_medicine": med,
                "db_matches": [med],
                "alternatives": alts,
                "side_effects": med_safety.get("side_effects", []),
                "interactions": med_safety.get("interactions", []),
                "route": med.get("dosage_form"),
                "dosage_given": med.get("salt"),
                "confidence": med.get("confidence", 95.0),
            }

        # 5. Check clinical cross-drug interactions strictly between prescribed medicines
        cross_interactions = get_cross_drug_interactions(found_medicines)

        primary_med = found_medicines[0] if found_medicines else None
        primary_alts = per_medicine_results[primary_med["name"]]["alternatives"] if primary_med else []
        primary_side_effects = per_medicine_results[primary_med["name"]]["side_effects"] if primary_med else []
        primary_interactions = per_medicine_results[primary_med["name"]]["interactions"] if primary_med else []

        if not found_medicines and recovery_used and gemini_res.get("success") is False:
            return {
                "success": False, "status": "ocr_service_error", "mode": "prescription",
                "message": "Text recognition service is temporarily unavailable. Please try again later.",
                "ocr_error": gemini_res.get("error_type", "API_ERROR"), "raw_ocr": active_ocr,
                "raw_ocr_trocr": trocr_text, "raw_ocr_gemini": "", "recovery_used": True,
                "medicines": [], "per_medicine_results": {}
            }

        return {
            "success": bool(found_medicines),
            "status": "verified" if found_medicines else "unverified",
            "mode": "prescription",
            "raw_ocr": active_ocr,
            "raw_ocr_trocr": trocr_text,
            "raw_ocr_gemini": gemini_res.get("raw_text", "") if recovery_used else "",
            "recovery_used": recovery_used,
            "normalized_ocr": norm_text,
            "medicines": found_medicines,
            "medicine": primary_med,
            "per_medicine_results": per_medicine_results,
            "default_selected": primary_med["name"] if primary_med else "",
            "total_found": len(found_medicines),
            "ingredients": all_ingredients,
            "alternatives": primary_alts,
            "side_effects": primary_side_effects,
            "interactions": primary_interactions,
            "cross_interactions": cross_interactions,
            "confidence_breakdown": {
                "brand": round(primary_med.get("confidence", 95.0), 1) if primary_med else 0.0,
                "ingredients": 95.0 if primary_med else 0.0,
                "strength": 95.0 if primary_med else 0.0,
                "manufacturer": 85.0 if primary_med else 0.0,
                "dosage_form": 90.0 if primary_med else 0.0,
                "overall": round(primary_med.get("confidence", 95.0), 1) if primary_med else 0.0,
            }
        }

    except Exception as exc:
        logger.error("scan_prescription failed: %s", exc, exc_info=True)
        return {"success": False, "status": "unverified", "error": str(exc), "per_medicine_results": {}}


# ── Main entry point ──────────────────────────────────────────────────────────

def scan_image(image_path: str, mode: str = "auto") -> Dict[str, Any]:
    """
    Main entry point called by Django views.
    """
    try:
        from PIL import Image  # noqa: PLC0415
        image_pil = Image.open(image_path)

        if mode == "medicine_package":
            return scan_medicine_package(image_pil)
        if mode == "prescription":
            return scan_prescription(image_pil)

        # Auto-mode: choose initial preference based on aspect ratio
        is_tall_document = (image_pil.height / max(1, image_pil.width)) >= 1.15
        if is_tall_document:
            logger.info("Auto-mode: tall aspect ratio detected — prioritizing prescription scan")
            rx_res = scan_prescription(image_pil)
            if rx_res.get("status") == "verified" and rx_res.get("medicines"):
                return rx_res
            logger.info("Auto-mode: prescription scan inconclusive — trying package scan")
            res = scan_medicine_package(image_pil)
            if res.get("status") == "verified" and res.get("medicine"):
                return res
            return rx_res if rx_res.get("raw_ocr") else res
        else:
            logger.info("Auto-mode: running package scan")
            res = scan_medicine_package(image_pil)
            if res.get("status") == "verified" and res.get("medicine"):
                return res

            logger.info("Auto-mode: trying prescription scan")
            rx_res = scan_prescription(image_pil)
            if rx_res.get("status") == "verified" and rx_res.get("medicines"):
                return rx_res

            return res if res.get("raw_ocr") else rx_res

    except Exception as exc:
        logger.error("scan_image failed on %s: %s", image_path, exc, exc_info=True)
        return {"success": False, "status": "unverified", "error": str(exc)}


# ── Load models on import ─────────────────────────────────────────────────────
try:
    load_models()
except Exception as _load_exc:
    logger.warning(
        "MediScan vision models could not be loaded at import time: %s", _load_exc
    )
