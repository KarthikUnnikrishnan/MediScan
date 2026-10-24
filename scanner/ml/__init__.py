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
        import time
        from PIL import ImageOps
        from .recognisers import get_printed_recogniser
        from .medicine_matcher import match_printed_lines
        from .salt_parser import parse_salt
        from .alternative_matcher import find_top_cheaper_alternatives
        from .drug_safety import get_drug_safety_info
        from django.conf import settings

        target_img = ImageOps.exif_transpose(image_pil)
        use_crop = getattr(settings, 'USE_STRIP_CROP', False)
        if use_crop:
            try:
                from .vision import crop_strip
                cropped, hit = crop_strip(target_img)
                if hit and cropped:
                    target_img = cropped
            except Exception as e:
                logger.debug("Strip crop skipped: %s", e)

        rec_a = get_printed_recogniser()

        # FIX 3.4: Early-exit rotation fallback & 15s time budget
        # Try 0, then 180; only try 90/270 if the first two give no confident result
        # and fewer than 2 tokens match known words. Set a total time budget of 15 s per photo.
        t_start = time.time()
        TIME_BUDGET_SEC = 15.0

        chosen_angle = 0
        chosen_res = None
        chosen_lines = []
        rot_candidates = []
        max_matching_tokens = 0

        # Phase 1: Try 0, then 180
        for angle in [0, 180]:
            if (time.time() - t_start) >= TIME_BUDGET_SEC:
                break
            rot_img = target_img.rotate(angle, expand=True) if angle != 0 else target_img
            lines = rec_a.read_lines(rot_img, beams=1)
            res = match_printed_lines(lines)

            top_cand = res.get("candidates") and res["candidates"][0]
            top_score = top_cand.get("score", 0.0) if top_cand else 0.0
            top_conf = bool(top_cand and top_cand.get("confident"))
            n_tokens = res.get("n_matching_tokens", 0)
            max_matching_tokens = max(max_matching_tokens, n_tokens)

            is_confident = top_conf and (n_tokens >= 2)
            rot_candidates.append({
                "angle": angle,
                "lines": lines,
                "res": res,
                "top_score": top_score,
                "is_confident": is_confident
            })

            if is_confident:
                chosen_angle = angle
                chosen_res = res
                chosen_lines = lines
                break

        # Phase 2: Only try 90/270 if the first two give no confident result and fewer than 2 tokens match known words
        if chosen_res is None and max_matching_tokens < 2 and ((time.time() - t_start) < TIME_BUDGET_SEC):
            for angle in [90, 270]:
                if (time.time() - t_start) >= TIME_BUDGET_SEC:
                    break
                rot_img = target_img.rotate(angle, expand=True)
                lines = rec_a.read_lines(rot_img, beams=1)
                res = match_printed_lines(lines)

                top_cand = res.get("candidates") and res["candidates"][0]
                top_score = top_cand.get("score", 0.0) if top_cand else 0.0
                top_conf = bool(top_cand and top_cand.get("confident"))
                n_tokens = res.get("n_matching_tokens", 0)

                is_confident = top_conf and (n_tokens >= 2)
                rot_candidates.append({
                    "angle": angle,
                    "lines": lines,
                    "res": res,
                    "top_score": top_score,
                    "is_confident": is_confident
                })

                if is_confident:
                    chosen_angle = angle
                    chosen_res = res
                    chosen_lines = lines
                    break

        if chosen_res is None and rot_candidates:
            best_rot = max(rot_candidates, key=lambda x: x["top_score"])
            chosen_angle = best_rot["angle"]
            chosen_res = best_rot["res"]
            chosen_lines = best_rot["lines"]

        match_res = chosen_res
        lines = chosen_lines

        status = match_res.get("status", "unverified")
        match_type = match_res.get("match_type", "brand")
        candidates = match_res.get("candidates", [])
        raw_ocr = match_res.get("raw_ocr", " ".join(lines))
        confidence_breakdown = match_res.get("confidence_breakdown", {
            "brand": 0.0, "ingredients": 0.0, "strength": 0.0,
            "manufacturer": 0.0, "dosage_form": 0.0, "overall": 0.0
        })

        if not candidates:
            return {
                "success": False,
                "status": "not_found",
                "result_level": "not_found",
                "match_type": match_type,
                "mode": "medicine_package",
                "rotation": chosen_angle,
                "message": "Could not identify this medicine. Try a clearer photo of the brand name.",
                "raw_ocr": raw_ocr,
                "medicine": None,
                "candidates": [],
                "medicines": [],
                "alternatives": [],
                "side_effects": [],
                "interactions": [],
                "confidence_breakdown": confidence_breakdown,
            }

        top_cand = candidates[0]
        # Step 7 Rule 4 check: exact brand AND salt matched
        is_exact_brand = bool(top_cand.get("is_exact_brand"))
        salt_matched = bool(top_cand.get("salt_matched"))
        is_rule4 = is_exact_brand and salt_matched

        result_level = "verified" if is_rule4 else "likely"
        status = "verified" if is_rule4 else "likely"

        verified_med = top_cand
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

        # Step 7 Result Wording
        if result_level == "verified":
            level_badge = "Verified match"
            level_text = "Matches the brand and the ingredient printed on the pack."
        else:
            level_badge = "Likely match - please confirm"
            if top_cand.get("match_type") == "ingredient":
                level_text = "Closest match by ingredient"
            else:
                level_text = "Top candidate based on packaging text — please confirm brand and strength."

        return {
            "success": True,
            "status": status,
            "result_level": result_level,
            "level_badge": level_badge,
            "level_text": level_text,
            "pack_warning": "Check this result against your pack before use.",
            "match_type": top_cand.get("match_type", match_type),
            "mode": "medicine_package",
            "rotation": chosen_angle,
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
                "is_exact_brand": is_exact_brand,
                "salt_matched": salt_matched,
                "match_type": top_cand.get("match_type", match_type),
            },
            "candidates": candidates[:3],
            "top_candidates": candidates[:3],
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
            "status": "error",
            "result_level": "error",
            "error_type": "setup_error",
            "error_reason": str(exc),
            "message": f"Scanner is not ready: {exc}",
            "error": f"Scanner is not ready: {exc}",
            "raw_ocr": "",
            "rotation": 0,
            "candidates": [],
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


ENGLISH_STOPWORDS = {
    'WAS', 'THE', 'AND', 'FOR', 'ARE', 'NOT', 'YOU', 'ALL', 'CAN', 'HAS', 'HAD', 'HAVE',
    'WITH', 'FROM', 'THIS', 'THAT', 'WHO', 'HIS', 'HER', 'SHE', 'HIM', 'BUT', 'OUT', 'DAY', 'DAYS'
}


def match_word(text: str) -> List[Dict[str, Any]]:
    """
    Calls the EXISTING brand-word matcher (BrandWordIndex / rapidfuzz) for ONE word (or short phrase).
    Returns list of [{'name': ..., 'score': 0-100, 'brand': ..., 'salt': ..., 'manufacturer': ..., 'price': ..., 'id': ...}]
    ranked best first. Does not modify Model A's behavior.
    """
    if not text or not text.strip():
        return []
    clean = re.sub(r'[^A-Za-z0-9]', '', text).upper()
    if len(clean) < 3:
        return []

    from .medicine_matcher import PHARMA_STOPWORDS
    from .handwritten_trocr import RX_STOP_WORDS, is_name_or_header_line
    if is_name_or_header_line(text):
        return []
    clean_lower = clean.lower()
    if clean in PHARMA_STOPWORDS or clean_lower in RX_STOP_WORDS or _RX_METADATA_IGNORE.search(text):
        return []
    toks = [re.sub(r'[^a-zA-Z0-9]', '', t).lower() for t in re.split(r'[\s/.,:;-]+', text)]
    if any(t in RX_STOP_WORDS for t in toks if t):
        return []

    from rapidfuzz import process, fuzz
    from .medicine_index import get_brand_word_index
    brand_index = get_brand_word_index()
    if not brand_index.is_loaded:
        brand_index.build()

    exact_match = clean in brand_index.brand_to_ids
    candidates = []
    seen_brands = set()

    if exact_match:
        seen_brands.add(clean)
        mids = brand_index.brand_to_ids.get(clean, [])[:3]
        rows = brand_index.get_medicines_by_ids(mids)
        for r in rows:
            candidates.append({
                'name': r['name'],
                'score': 100.0,
                'brand': clean,
                'salt': r['salt'] or '',
                'manufacturer': r['manufacturer'] or 'Registered Manufacturer',
                'price': r['price'],
                'id': r['id'],
            })

    f_matches = process.extract(
        clean,
        brand_index.unique_brand_words,
        scorer=fuzz.ratio,
        limit=5,
        score_cutoff=60.0
    )
    for b_word, score, _ in f_matches:
        if b_word in seen_brands:
            continue
        if len(b_word) < 3 or (len(clean) <= 4 and abs(len(clean) - len(b_word)) > 1):
            continue
        seen_brands.add(b_word)
        mids = brand_index.brand_to_ids.get(b_word, [])[:2]
        rows = brand_index.get_medicines_by_ids(mids)
        for r in rows:
            candidates.append({
                'name': r['name'],
                'score': round(float(score), 1),
                'brand': b_word,
                'salt': r['salt'] or '',
                'manufacturer': r['manufacturer'] or 'Registered Manufacturer',
                'price': r['price'],
                'id': r['id'],
            })

    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates


# ── Prescription scan (100% Offline Model Driven) ─────────────────────────────

def scan_prescription(image_pil) -> Dict[str, Any]:
    """
    Execute prescription scan workflow using Model B (HandwrittenTrOCR) and local SQLite validation.
    Returns structured result with per-medicine entries, candidates, dosage instructions,
    safety data, and bioequivalent cheaper alternatives. 100% offline.
    """
    try:
        from .recognisers import get_handwritten_recogniser
        rec_b = get_handwritten_recogniser()
        res_b = rec_b.read_prescription(image_pil, matcher=match_word, beams=2, min_score=80.0)

        timing = res_b.get("timing", {"detect": 0.0, "read": 0.0, "match": 0.0, "total": 0.0})
        other_text = res_b.get("other_text", [])

        if not res_b.get("ok"):
            msg = res_b.get("message") or "Handwritten model is not installed yet"
            return {
                "success": False,
                "status": "error",
                "result_level": "error",
                "error_type": "setup_error",
                "error_reason": msg,
                "mode": "prescription",
                "message": f"Scanner is not ready: {msg}",
                "error": f"Scanner is not ready: {msg}",
                "raw_ocr": "",
                "ocr_text": "",
                "other_text_read": other_text,
                "candidates": [],
                "medicines": [],
                "per_medicine_results": {},
                "timing": timing,
                "disclaimer": "Handwriting can be misread - confirm every medicine with your doctor or pharmacist",
            }

        lines = res_b.get("lines", [])
        raw_ocr_lines = [l["text"] for l in lines if l.get("text")]
        raw_ocr = "\n".join(raw_ocr_lines)

        meds = res_b.get("medicines", [])
        matched_meds = [m for m in meds if m.get("candidates") and m["candidates"][0].get("score", 0) >= 80.0]

        if not matched_meds:
            return {
                "success": False,
                "status": "not_found",
                "result_level": "not_found",
                "mode": "prescription",
                "message": "Could not identify this medicine. Try a clearer photo of the brand name.",
                "raw_ocr": raw_ocr,
                "ocr_text": raw_ocr,
                "other_text_read": other_text or raw_ocr_lines,
                "candidates": [],
                "medicines": [],
                "per_medicine_results": {},
                "timing": timing,
                "disclaimer": "Handwriting can be misread - confirm every medicine with your doctor or pharmacist",
            }

        per_medicine_results: Dict[str, Any] = {}
        medicines_list: List[Dict[str, Any]] = []
        all_salts_for_interaction: List[List[Dict[str, Any]]] = []

        for p in matched_meds:
            top_cand = p["candidates"][0]
            med_key = top_cand.get("brand") or top_cand.get("name", "").split()[0]
            if med_key in per_medicine_results:
                med_key = f"{med_key} ({p.get('dose') or len(per_medicine_results)+1})"

            salt_str = top_cand.get("salt", "")
            parsed_salt = parse_salt(salt_str) if salt_str else []
            top_cand["parsed_salt"] = parsed_salt
            if parsed_salt:
                all_salts_for_interaction.append(parsed_salt)

            safety_info = get_drug_safety_info(parsed_salt)
            alternatives = find_top_cheaper_alternatives(top_cand, limit=5)

            med_entry = {
                "line": p.get("line", 0),
                "text_read": p.get("read", ""),
                "read": p.get("read", ""),
                "text": p.get("text", ""),
                "form": p.get("form"),
                "dosage_form": p.get("form") or top_cand.get("dosage_form", "Tablet"),
                "dose": p.get("dose"),
                "dosage_given": p.get("dose"),
                "frequency": p.get("frequency"),
                "duration": p.get("duration"),
                "confidence": top_cand.get("score", 80.0),
                "top_medicine": top_cand,
                "candidates": p.get("candidates", [])[:3],
                "top_3_candidates": p.get("candidates", [])[:3],
                "db_matches": p.get("candidates", [])[:3],
                "alternatives": alternatives,
                "side_effects": safety_info.get("side_effects", []),
                "interactions": safety_info.get("interactions", []),
            }

            per_medicine_results[med_key] = med_entry
            medicines_list.append(top_cand)

        cross_interactions = []
        if len(all_salts_for_interaction) >= 2:
            try:
                cross_interactions = get_cross_drug_interactions(all_salts_for_interaction)
            except Exception as e:
                logger.debug("Cross interaction lookup error: %s", e)

        first_key = next(iter(per_medicine_results.keys())) if per_medicine_results else ""
        first_alts = per_medicine_results[first_key]["alternatives"] if first_key else []

        top_med = medicines_list[0] if medicines_list else {}
        is_exact_brand = bool(top_med.get("is_exact_brand"))
        salt_matched = bool(top_med.get("salt_matched"))
        is_rule4 = is_exact_brand and salt_matched
        result_level = "verified" if is_rule4 else "likely"

        return {
            "success": True,
            "status": "verified" if is_rule4 else "likely",
            "result_level": result_level,
            "level_badge": "Verified match" if is_rule4 else "Likely match - please confirm",
            "level_text": "Matches the brand and the ingredient printed on the pack." if is_rule4 else ("Closest match by ingredient" if top_med.get("match_type") == "ingredient" else "Likely match — please confirm"),
            "pack_warning": "Check this result against your pack before use.",
            "mode": "prescription",
            "message": "Prescription parsed successfully.",
            "raw_ocr": raw_ocr,
            "ocr_text": raw_ocr,
            "other_text_read": other_text,
            "detected_name": first_key,
            "medicine": medicines_list[0] if medicines_list else None,
            "medicines": medicines_list,
            "candidates": medicines_list[:3],
            "alternatives": first_alts,
            "cross_interactions": cross_interactions,
            "per_medicine_results": per_medicine_results,
            "total_found": len(per_medicine_results),
            "default_selected": first_key,
            "timing": timing,
            "disclaimer": "Handwriting can be misread - confirm every medicine with your doctor or pharmacist",
        }
    except Exception as exc:
        logger.error("scan_prescription failed: %s", exc, exc_info=True)
        return {
            "success": False,
            "status": "error",
            "result_level": "error",
            "error_type": "setup_error",
            "error_reason": str(exc),
            "mode": "prescription",
            "message": f"Scanner is not ready: {exc}",
            "error": f"Scanner is not ready: {exc}",
            "candidates": [],
            "medicines": [],
            "per_medicine_results": {},
            "raw_ocr": "",
            "ocr_text": "",
            "other_text_read": [],
            "timing": {"detect": 0.0, "read": 0.0, "match": 0.0, "total": 0.0},
            "disclaimer": "Handwriting can be misread - confirm every medicine with your doctor or pharmacist",
        }


# ── Main entry point ──────────────────────────────────────────────────────────

def scan_image(image_path: str, mode: str = "medicine_package") -> Dict[str, Any]:
    """
    Main entry point called by Django views.
    Executes 100% offline with separate models per mode.
    """
    try:
        from PIL import Image, ImageOps
        image_pil = Image.open(image_path)
        image_pil = ImageOps.exif_transpose(image_pil)
        if mode == "prescription":
            return scan_prescription(image_pil)
        return scan_medicine_package(image_pil)
    except Exception as exc:
        logger.error("scan_image failed on %s: %s", image_path, exc, exc_info=True)
        return {
            "success": False,
            "status": "error",
            "result_level": "error",
            "error_type": "setup_error",
            "error_reason": str(exc),
            "message": f"Scanner is not ready: {exc}",
            "error": f"Scanner is not ready: {exc}",
            "candidates": [],
            "medicines": []
        }




