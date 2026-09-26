"""
scanner/ml/medicine_matcher.py — Precision medicine identification engine.

Implements multi-tier matching with hard composition validation,
ingredient-by-ingredient strength verification, dosage form compatibility,
and real confidence scoring.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from rapidfuzz import fuzz, process

from .salt_parser import parse_salt, compare_compositions, normalize_ingredient_name
from .medicine_index import get_medicine_index, MedicineIndex

logger = logging.getLogger(__name__)

# Minimum overall confidence required to consider a medicine verified
CONFIDENCE_THRESHOLD = 70.0


def are_forms_compatible(form_a: Optional[str], form_b: Optional[str]) -> Tuple[bool, float]:
    """
    Check if two dosage forms are clinically compatible.
    Returns (compatible_bool, compatibility_score_0_to_100).
    """
    if not form_a or not form_b:
        return True, 70.0  # Neutral if one is unknown

    fa = form_a.lower()
    fb = form_b.lower()

    if fa == fb:
        return True, 100.0

    # Solid oral forms
    solid_oral = {'tablet', 'capsule', 'caplet', 'dt', 'dispersible'}
    if fa in solid_oral and fb in solid_oral:
        return True, 85.0

    # Liquid oral forms
    liquid_oral = {'syrup', 'suspension', 'liquid', 'solution', 'elixir'}
    if fa in liquid_oral and fb in liquid_oral:
        return True, 85.0

    # Topical forms
    topical = {'ointment', 'gel', 'cream', 'lotion'}
    if fa in topical and fb in topical:
        return True, 80.0

    # Strictly incompatible (e.g. Tablet vs Syrup, Injection vs Tablet)
    return False, 0.0


def match_medicine(extracted: Dict[str, Any], index: Optional[MedicineIndex] = None) -> Dict[str, Any]:
    """
    Match extracted OCR information against the dynamic medicine index.

    Parameters
    ----------
    extracted : dict
        {
            "brand": str or None,
            "ingredients": list of dict,
            "manufacturer": str or None,
            "dosage_form": str or None,
            "raw_ocr": str,
            "normalized_ocr": str
        }
    index : MedicineIndex, optional

    Returns
    -------
    dict:
        {
            "status": "verified" | "unverified",
            "medicine": dict or None,
            "confidence_breakdown": {
                "brand": float,
                "ingredients": float,
                "strength": float,
                "manufacturer": float,
                "dosage_form": float,
                "overall": float
            },
            "candidate_count": int,
            "rejection_reasons": list of str,
            "validation_details": dict
        }
    """
    if index is None:
        index = get_medicine_index()

    brand_query = extracted.get("brand") or ""
    detected_ingredients = extracted.get("ingredients") or []
    detected_mfr = extracted.get("manufacturer")
    detected_form = extracted.get("dosage_form")
    normalized_ocr = extracted.get("normalized_ocr") or ""

    candidates: List[Dict[str, Any]] = []
    seen_ids = set()

    def add_candidate(med: Dict[str, Any]):
        if med["id"] not in seen_ids:
            seen_ids.add(med["id"])
            candidates.append(med)

    # ── 1. Exact normalized brand & dynamic alias lookup ───────────────────────
    if brand_query:
        for med in index.search_alias(brand_query):
            add_candidate(med)

    # ── 2. Brand + Strength search ─────────────────────────────────────────────
    # Extract numbers from OCR to search e.g. "Dolo 650", "Augmentin 625"
    numbers = re.findall(r'\b\d+\b', normalized_ocr)
    if brand_query:
        first_word = brand_query.split()[0]
        for num in numbers:
            combos = [f"{first_word} {num}", f"{first_word}{num}"]
            for c in combos:
                for med in index.search_alias(c):
                    add_candidate(med)

    # ── 3. Search by exact composition key if ingredients extracted ────────────
    if detected_ingredients:
        # Build key from detected ingredients
        sorted_ing = sorted(detected_ingredients, key=lambda x: x.get('canonical_name', ''))
        comp_key = "+".join([
            f"{ing.get('canonical_name', '')}:{ing.get('strength', '')}:{ing.get('unit', '')}"
            for ing in sorted_ing
        ])
        for med in index.find_by_composition_key(comp_key):
            add_candidate(med)

    # ── 4. RapidFuzz fallback on brand names if candidate pool is small ────────
    if len(candidates) < 5 and brand_query:
        first_brand_word = brand_query.split()[0].lower()
        if len(first_brand_word) >= 3:
            fuzzy_matches = process.extract(
                first_brand_word,
                index.unique_brand_names,
                scorer=fuzz.token_sort_ratio,
                limit=3,
                score_cutoff=82
            )
            for match_name, score, _ in fuzzy_matches:
                for med in index.search_alias(match_name):
                    add_candidate(med)

    if not candidates:
        return {
            "status": "unverified",
            "medicine": None,
            "confidence_breakdown": {
                "brand": 0.0, "ingredients": 0.0, "strength": 0.0,
                "manufacturer": 0.0, "dosage_form": 0.0, "overall": 0.0
            },
            "candidate_count": 0,
            "rejection_reasons": ["No matching brand, alias, or composition found in database."],
            "validation_details": {}
        }

    # ── 5. Score and validate candidates ───────────────────────────────────────
    scored_candidates = []

    for cand in candidates:
        rejection_reasons = []
        cand_name = cand.get("name", "")
        cand_salts = cand.get("parsed_salt") or []
        cand_mfr = cand.get("manufacturer") or ""
        cand_form = cand.get("dosage_form")

        # a. Brand score
        brand_score = 0.0
        if brand_query:
            b_score1 = fuzz.token_set_ratio(brand_query.lower(), cand_name.lower())
            b_score2 = fuzz.partial_ratio(brand_query.lower(), cand_name.lower())
            brand_score = max(b_score1, b_score2)
        else:
            brand_score = 50.0

        # b. Hard composition validation (Critical for combination drugs!)
        composition_valid = True
        ing_score = 0.0
        strength_score = 0.0

        if detected_ingredients and cand_salts:
            det_canons = {ing["canonical_name"] for ing in detected_ingredients if ing.get("canonical_name")}
            cand_canons = {ing["canonical_name"] for ing in cand_salts if ing.get("canonical_name")}

            # If detected was a combination drug, require all ingredients to match!
            if len(detected_ingredients) > 1:
                if det_canons != cand_canons:
                    composition_valid = False
                    rejection_reasons.append(
                        f"Combination mismatch: detected {sorted(det_canons)} vs candidate {sorted(cand_canons)}"
                    )
                else:
                    ing_score = 100.0
            else:
                # Single detected ingredient
                if det_canons == cand_canons:
                    ing_score = 100.0
                elif det_canons.issubset(cand_canons) and len(cand_canons) > len(det_canons):
                    # Candidate has extra ingredients not detected
                    composition_valid = False
                    rejection_reasons.append(
                        f"Candidate contains un-detected active ingredients: {sorted(cand_canons - det_canons)}"
                    )
                else:
                    ing_score = 0.0
                    composition_valid = False

            # c. Strength verification (ingredient-by-ingredient)
            if composition_valid:
                strength_matches = 0
                for d_ing in detected_ingredients:
                    d_cname = d_ing.get("canonical_name")
                    d_st = d_ing.get("strength")
                    for c_ing in cand_salts:
                        if c_ing.get("canonical_name") == d_cname:
                            if c_ing.get("strength") == d_st:
                                strength_matches += 1
                if strength_matches == len(detected_ingredients) and len(detected_ingredients) > 0:
                    strength_score = 100.0
                elif len(detected_ingredients) > 0:
                    strength_score = (strength_matches / len(detected_ingredients)) * 100.0
                    # Reject candidate if strengths conflict
                    if strength_matches < len(detected_ingredients):
                        rejection_reasons.append("Strength mismatch between detected and candidate.")
        else:
            # If OCR did not detect separate ingredient lines (e.g. blister pack only has brand like Dolo 650)
            if cand_salts:
                ing_score = 75.0
                strength_score = 75.0
            else:
                # Scraped row with corrupted or missing salt
                ing_score = 40.0
                strength_score = 40.0

        # Disambiguate strength using brand name context (e.g. Dolo 650)
        brand_numbers = re.findall(r'\b\d+\b', brand_query)
        if brand_numbers:
            target_num = brand_numbers[0]
            if target_num in cand_name:
                strength_score = max(strength_score, 95.0)
            else:
                strength_score = min(strength_score, 40.0)

        # d. Dosage form score
        form_compatible, form_score = are_forms_compatible(detected_form, cand_form)
        if not form_compatible:
            rejection_reasons.append(f"Incompatible dosage form: detected {detected_form} vs {cand_form}")

        # e. Manufacturer score
        mfr_score = 70.0
        if detected_mfr and cand_mfr:
            mfr_ratio = fuzz.token_set_ratio(detected_mfr.lower(), cand_mfr.lower())
            mfr_score = float(mfr_ratio)

        # Compute overall confidence
        if detected_ingredients:
            # If detected ingredients exist but candidate has empty salt, penalize heavily
            if not cand_salts:
                ing_score = 20.0
                strength_score = 20.0
            overall = (
                (brand_score * 0.35) +
                (ing_score * 0.30) +
                (strength_score * 0.20) +
                (form_score * 0.10) +
                (mfr_score * 0.05)
            )
        else:
            overall = (
                (brand_score * 0.55) +
                (strength_score * 0.25) +
                (form_score * 0.15) +
                (mfr_score * 0.05)
            )

        # Quality bonus for verified rows with price, manufacturer, and valid salt
        overall += cand.get("quality_score", 0) * 0.8

        if not composition_valid or not form_compatible:
            overall = min(overall, 45.0)

        scored_candidates.append({
            "candidate": cand,
            "brand_score": round(brand_score, 1),
            "ingredient_score": round(ing_score, 1),
            "strength_score": round(strength_score, 1),
            "manufacturer_score": round(mfr_score, 1),
            "dosage_form_score": round(form_score, 1),
            "overall_confidence": round(overall, 1),
            "rejection_reasons": rejection_reasons,
            "composition_valid": composition_valid
        })

    # Sort candidates by overall confidence descending
    scored_candidates.sort(key=lambda x: x["overall_confidence"], reverse=True)
    top_cand = scored_candidates[0]

    def _cap(val: float) -> float:
        return min(100.0, max(0.0, round(val, 1)))

    breakdown = {
        "brand": _cap(top_cand["brand_score"]),
        "ingredients": _cap(top_cand["ingredient_score"]),
        "strength": _cap(top_cand["strength_score"]),
        "manufacturer": _cap(top_cand["manufacturer_score"]),
        "dosage_form": _cap(top_cand["dosage_form_score"]),
        "overall": _cap(top_cand["overall_confidence"])
    }

    if breakdown["overall"] < CONFIDENCE_THRESHOLD or not top_cand["composition_valid"]:
        logger.info(
            "Top candidate %s rejected (confidence %.1f < threshold %.1f)",
            top_cand["candidate"]["name"], breakdown["overall"], CONFIDENCE_THRESHOLD
        )
        return {
            "status": "unverified",
            "medicine": None,
            "confidence_breakdown": breakdown,
            "candidate_count": len(scored_candidates),
            "rejection_reasons": top_cand["rejection_reasons"] or ["Confidence below verification threshold."],
            "validation_details": top_cand
        }

    selected_med = top_cand["candidate"]
    return {
        "status": "verified",
        "medicine": selected_med,
        "confidence_breakdown": breakdown,
        "candidate_count": len(scored_candidates),
        "rejection_reasons": [],
        "validation_details": top_cand
    }
