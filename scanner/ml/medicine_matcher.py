"""
scanner/ml/medicine_matcher.py — Precision medicine identification engine.

Implements multi-tier matching with hard composition validation,
ingredient-by-ingredient strength verification, dosage form compatibility,
and real confidence scoring.
"""

import re
import math
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from rapidfuzz import fuzz, process

from .salt_parser import (
    parse_salt,
    compare_compositions,
    normalize_ingredient_name,
    extract_strengths_from_text,
    extract_active_salt_set,
    _KNOWN_ACTIVE_NAMES,
)
from .medicine_index import get_medicine_index, MedicineIndex

logger = logging.getLogger(__name__)

# Minimum overall confidence required to consider a medicine verified
CONFIDENCE_THRESHOLD = 70.0

BOILERPLATE_BOX_PATTERNS = [
    r'\b(?:DIRECTED|PHYSICIAN|DOSAGE|STORAGE|STORE|TEMPERATURE|PROTECT|MOISTURE|LIGHT)\b',
    r'\b(?:KEEP\s+OUT|REACH\s+OF\s+CHILDREN|NOT\s+TO\s+BE)\b',
    r'\b(?:SCHEDULE\s+[HGHX1]+|PRESCRIPTION\s+DRUG|WARNING|CAUTION|RETAIL)\b',
    r'\b(?:REGISTERED\s+MEDICAL|PRACTITIONER)\b',
    r'\b(?:MANUFACTURED|MFG|BATCH|EXP|DATE|MKT|MARKETED|PACKED|LIC\s*NO)\b',
    r'\b(?:EACH\s+(?:FILM|UNCOATED|COATED)?\s*TABLET|CONTAINS)\b',
    r'\b(?:COLOUR|EXCIPIENTS|APPROPRIATE\s+OVERAGES)\b',
    r'\b(?:LIMITED|LTD|PVT|PHARMACEUTICALS?|LABORATORIES|LABS?)\b',
]

GRAMMAR_WORDS = {
    'BY', 'THE', 'OF', 'FOR', 'IN', 'TO', 'AS', 'AND', 'WITH', 'FROM',
    'AT', 'ON', 'OUT', 'NOT', 'BE', 'AN', 'IS', 'OR'
}

COMMON_FRAGMENTS = {
    'sted', 'cted', 'ated', 'nted', 'rted', 'lted', 'pted', 'fted', 'kted',
    'tion', 'sion', 'ment', 'ance', 'ence', 'able', 'ible', 'less', 'ness',
    'ship', 'ward', 'wise', 'like', 'some', 'full', 'most', 'ever',
    'take', 'used', 'oral', 'keep', 'from', 'with', 'pack', 'dose', 'unit',
    'phys', 'cian', 'cial', 'tial', 'rect', 'each', 'film', 'coat', 'tabs',
}


def is_brand_box_candidate(line_str: str) -> bool:
    """
    Check if a recognized OCR line looks like a prominent brand box rather
    than a sentence, instructional text, or composition/dosage print.
    """
    if not line_str or not line_str.strip():
        return False
    u = line_str.upper()
    for pat in BOILERPLATE_BOX_PATTERNS:
        if re.search(pat, u):
            return False
    words = [w for w in re.split(r'[\s\-_]+', u) if w]
    if any(w in GRAMMAR_WORDS for w in words):
        return False
    if len(words) > 3:
        return False
    return True


def _zero_breakdown() -> Dict[str, float]:
    return {
        "brand": 0.0, "ingredients": 0.0, "strength": 0.0,
        "manufacturer": 0.0, "dosage_form": 0.0, "overall": 0.0
    }


def _brand_has_strength_signal(text: str) -> bool:
    """Return whether text contains a dosage value, not an arbitrary OCR digit."""
    return bool(re.search(
        r'\b\d+(?:\.\d+)?\s*(?:mg|ml|mcg|g|iu)\b',
        text or "",
        re.IGNORECASE,
    ))


def _extract_strength_numbers(brand: str, normalized_ocr: str) -> List[str]:
    """
    Extract numbers that can identify a medicine strength.

    Unitless values are accepted only when they are part of the extracted
    brand (for example, ``Dolo 650``). Standalone OCR digits such as a
    blister count or a hallucinated trailing token are not strength evidence.
    """
    numbers = re.findall(
        r'\b(\d+(?:\.\d+)?)\s*(?:mg|ml|mcg|g|iu|%)(?!\w)',
        f"{brand} {normalized_ocr}",
        re.IGNORECASE,
    )
    strength_numbers = list(numbers)

    brand_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', brand or "")
    for number in brand_numbers:
        if number not in strength_numbers:
            strength_numbers.append(number)
    return strength_numbers


def _normalize_form(form: Optional[str]) -> str:
    if not form:
        return ""
    f = form.lower().strip()
    f = re.sub(r'[^a-z]', '', f)
    if f.endswith('s') and f[:-1] in {'tablet', 'capsule', 'caplet', 'syrup', 'suspension', 'injection', 'drop', 'lotion', 'cream', 'gel', 'ointment', 'patch', 'respule'}:
        f = f[:-1]
    if f == 'drop':
        f = 'drops'
    return f


def are_forms_compatible(form_a: Optional[str], form_b: Optional[str]) -> Tuple[bool, float]:
    """
    Check if two dosage forms are clinically compatible.
    Returns (compatible_bool, compatibility_score_0_to_100).
    """
    if not form_a or not form_b:
        return True, 70.0  # Neutral if one is unknown

    fa = _normalize_form(form_a)
    fb = _normalize_form(form_b)

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


def match_medicine(
    extracted: Dict[str, Any],
    index: Optional[MedicineIndex] = None,
    is_prescription: bool = False,
) -> Dict[str, Any]:
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
    is_prescription : bool, optional
        Set to True when scanning prescriptions where doctors specify brand + dosage form
        without packaging-only manufacturer or salt details.

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

    if extracted.get("is_prescription"):
        is_prescription = True

    brand_query = extracted.get("brand") or ""
    detected_ingredients = extracted.get("ingredients") or []
    detected_mfr = extracted.get("manufacturer")
    detected_form = extracted.get("dosage_form")
    normalized_ocr = extracted.get("normalized_ocr") or ""

    has_ingredient_evidence = bool(detected_ingredients)
    strength_numbers = _extract_strength_numbers(brand_query, normalized_ocr)
    has_brand_strength_evidence = bool(strength_numbers)
    has_manufacturer_evidence = bool(detected_mfr)
    if not has_ingredient_evidence and not brand_query:
        return {
            "status": "unverified",
            "medicine": None,
            "confidence_breakdown": _zero_breakdown(),
            "candidate_count": 0,
            "rejection_reasons": ["No usable brand or active-ingredient evidence extracted from OCR."],
            "validation_details": {}
        }

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
        # Search hyphen and spacing variants (e.g. "P-650" <-> "P 650" <-> "P650", "1-AL" <-> "1 AL")
        for alt_q in [
            brand_query.replace("-", " "),
            brand_query.replace(" ", "-"),
            brand_query.replace("-", ""),
            brand_query.replace(" ", ""),
        ]:
            if alt_q and alt_q != brand_query:
                for med in index.search_alias(alt_q):
                    add_candidate(med)
        brand_base = re.split(r'[-\s\d]+', brand_query)[0]
        if brand_base and len(brand_base) >= 3 and brand_base != brand_query:
            for med in index.search_alias(brand_base):
                add_candidate(med)

    # ── 2. Brand + Strength search ─────────────────────────────────────────────
    # Extract numbers from OCR to search e.g. "Dolo 650", "Augmentin 625", "Fexodin 120"
    if brand_query:
        first_word = re.split(r'[-\s]+', brand_query)[0]
        for num in strength_numbers:
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

        # If exact strength composition found few/no candidates, search by canonical ingredient names
        if len(candidates) < 5:
            det_canons = [ing.get('canonical_name') for ing in detected_ingredients if ing.get('canonical_name')]
            if det_canons:
                for k, mids in index.composition_to_ids.items():
                    if all(c in k for c in det_canons):
                        for mid in mids[:5]:
                            med = index.get_medicine(mid)
                            if med:
                                add_candidate(med)
                        if len(candidates) >= 20:
                            break

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
            "confidence_breakdown": _zero_breakdown(),
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
        if not cand_salts and cand.get("salt"):
            cand_salts = parse_salt(cand["salt"])
            if not cand_salts:
                cand_salts = parse_salt(cand_name)
        cand_mfr = cand.get("manufacturer") or ""
        cand_form = cand.get("dosage_form")

        # a. Brand score
        brand_score = 0.0
        if brand_query:
            clean_b = re.sub(r'[^a-z0-9]', '', brand_query.lower())
            clean_c = re.sub(r'[^a-z0-9]', '', cand_name.lower())

            # Exact compacted match or prefix (e.g. 'p650' in 'p650tablet' or '1al' in '1altablet')
            if clean_b and (clean_b == clean_c or clean_c.startswith(clean_b) or f"{clean_b}tablet" in clean_c or f"{clean_b}capsule" in clean_c):
                brand_score = 95.0
            else:
                b_words = [re.sub(r'[^a-z0-9]', '', w.lower()) for w in brand_query.split() if w.strip()]
                c_words = [re.sub(r'[^a-z0-9]', '', w.lower()) for w in cand_name.split() if w.strip()]
                first_b = b_words[0] if b_words else ""
                first_c = c_words[0] if c_words else ""

                if first_b and first_c and first_b == first_c:
                    brand_score = 95.0
                elif first_b and first_c and len(first_b) <= 2 and first_c != first_b:
                    # Short codes like 'p' or 'd' cannot match completely different words like 'atpera' or 'algina'
                    brand_score = 20.0
                elif first_b and first_c and fuzz.ratio(first_b, first_c) < 65:
                    brand_score = 30.0
                else:
                    b_score1 = fuzz.token_set_ratio(brand_query.lower(), cand_name.lower())
                    b_score2 = fuzz.partial_ratio(brand_query.lower(), cand_name.lower())
                    brand_score = max(b_score1, b_score2)
                    brand_base = re.sub(r'\b\d+(?:\.\d+)?\b', '', brand_query).strip()
                    if brand_base and len(brand_base) >= 3 and brand_base != brand_query:
                        b_score3 = fuzz.token_set_ratio(brand_base.lower(), cand_name.lower())
                        b_score4 = fuzz.partial_ratio(brand_base.lower(), cand_name.lower())
                        brand_score = max(brand_score, b_score3, b_score4)
        else:
            brand_score = 50.0

        # b. Hard composition validation (Critical for combination drugs!)
        composition_valid = True
        ing_score = 0.0
        strength_score = 0.0

        brand_resolved = False
        if detected_ingredients and cand_salts:
            det_canons = {ing["canonical_name"] for ing in detected_ingredients if ing.get("canonical_name")}
            cand_canons = {ing["canonical_name"] for ing in cand_salts if ing.get("canonical_name")}
            # Supplement candidate active composition if candidate name explicitly mentions the active ingredient
            cand_name_l = cand_name.lower()
            for d_c in det_canons:
                if d_c in cand_name_l:
                    cand_canons.add(d_c)

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
                    # Check if the single detected ingredient is actually the BRAND name with dosage
                    # (e.g. 'Lipitor 10mg' parsed as ingredient='lipitor', strength=10; candidate='Lipitor 10mg Tablet')
                    single_det_name = list(det_canons)[0]
                    cand_name_lower = cand_name.lower()
                    clean_cand_name = re.sub(r'[^a-z0-9]', ' ', cand_name_lower)
                    clean_single_det = re.sub(r'[^a-z0-9]', ' ', single_det_name.lower()).strip()
                    clean_brand_query = re.sub(r'[^a-z0-9]', ' ', brand_query.lower()).strip() if brand_query else ""
                    brand_match = (
                        (brand_score >= 75.0) and
                        (
                            clean_single_det in clean_cand_name
                            or single_det_name in cand_name_lower
                            or (clean_brand_query and clean_single_det in clean_brand_query)
                        )
                    )
                    if brand_match:
                        d_st = detected_ingredients[0].get("strength")
                        cand_strengths = [s.get("strength") for s in cand_salts if s.get("strength") is not None]
                        cand_nums = [float(n) for n in re.findall(r'\b\d+(?:\.\d+)?\b', cand_name)]
                        if d_st is not None and (d_st in cand_strengths or d_st in cand_nums):
                            ing_score = 95.0
                            strength_score = 100.0
                            composition_valid = True
                            brand_resolved = True
                        elif d_st is None:
                            ing_score = 90.0
                            strength_score = 85.0
                            composition_valid = True
                            brand_resolved = True
                        else:
                            ing_score = 0.0
                            composition_valid = False
                            rejection_reasons.append(
                                f"Strength mismatch: detected {d_st} vs candidate {cand_strengths}"
                            )
                    else:
                        ing_score = 0.0
                        composition_valid = False

            # c. Strength verification (ingredient-by-ingredient)
            if composition_valid and not brand_resolved:
                strength_matches = 0
                has_explicit_det_strength = any(d.get("strength") is not None for d in detected_ingredients)
                has_strength_conflict = False

                for d_ing in detected_ingredients:
                    d_cname = d_ing.get("canonical_name")
                    d_st = d_ing.get("strength")
                    for c_ing in cand_salts:
                        if c_ing.get("canonical_name") == d_cname:
                            c_st = c_ing.get("strength")
                            if d_st is not None and c_st is not None:
                                if d_st == c_st:
                                    strength_matches += 1
                                else:
                                    has_strength_conflict = True
                            elif d_st is None:
                                # Unquantified detected ingredient matches candidate formulation
                                strength_matches += 1

                if strength_matches == len(detected_ingredients) and len(detected_ingredients) > 0:
                    strength_score = 100.0 if has_explicit_det_strength else 85.0
                elif len(detected_ingredients) > 0:
                    strength_score = (strength_matches / len(detected_ingredients)) * 100.0
                    if has_strength_conflict:
                        composition_valid = False
                        rejection_reasons.append("Strength mismatch between detected and candidate.")
        else:
            # Brand-only OCR can rank candidates but must not masquerade as
            # composition verification. It requires strong brand evidence plus
            # either visible strength or manufacturer corroboration below.
            if cand_salts:
                ing_score = 0.0
                strength_score = 0.0
            else:
                ing_score = 0.0
                strength_score = 0.0
                rejection_reasons.append("Candidate has no parseable active composition.")

        # Disambiguate strength using brand name context (e.g. Dolo 650)
        brand_numbers = strength_numbers
        if brand_numbers:
            target_num = brand_numbers[0]
            if target_num in cand_name:
                strength_score = max(strength_score, 95.0)
            else:
                cand_salt_strengths = [
                    str(s.get('strength')) for s in cand_salts if s.get('strength') is not None
                ]
                cand_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', cand_name)
                if cand_salts and target_num in cand_salt_strengths:
                    # Candidate active composition corroborates this exact strength
                    strength_score = max(strength_score, 95.0)
                elif cand_numbers and target_num not in cand_numbers:
                    # Candidate name specifies a different number and composition does not match
                    strength_score = min(strength_score, 40.0)
                    composition_valid = False
                    rejection_reasons.append(
                        f"Visible strength/number {target_num} does not match candidate name."
                    )
                elif not cand_salts and target_num not in cand_name:
                    strength_score = min(strength_score, 40.0)
                    composition_valid = False
                    rejection_reasons.append(
                        f"Visible strength/number {target_num} does not match candidate name."
                    )

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
            has_form_evidence = bool(is_prescription and detected_form and form_compatible and form_score >= 70.0)
            if is_prescription and has_form_evidence and strength_score == 0:
                # In prescriptions, doctors frequently omit strength for standard brands/pediatric liquids (e.g. 'Syp MEFTAL-P')
                overall = (
                    (brand_score * 0.65) +
                    (form_score * 0.25) +
                    (mfr_score * 0.10)
                )
            else:
                overall = (
                    (brand_score * (0.50 if has_form_evidence else 0.55)) +
                    (strength_score * 0.25) +
                    (form_score * (0.20 if has_form_evidence else 0.15)) +
                    (mfr_score * 0.05)
                )

            if brand_score < 90.0:
                rejection_reasons.append(
                    "No active ingredient was extracted, and brand evidence is not exact enough for verification."
                )
                overall = min(overall, 55.0)

            if not has_brand_strength_evidence and not has_manufacturer_evidence and not has_form_evidence:
                rejection_reasons.append(
                    "No active ingredient, visible strength, or manufacturer corroboration was extracted."
                )
                overall = min(overall, 55.0)

            # If candidate name specifies a major strength number (e.g. 650 in 'Dolo 650 Tablet' or 500 in '500mg')
            # but OCR lacks that strength corroboration, do not verify based on form alone.
            cand_name_strengths = re.findall(r'(\d+(?:\.\d+)?)', cand_name)
            cand_has_major_strength = any(float(n) >= 10 for n in cand_name_strengths)
            if cand_has_major_strength and not (set(strength_numbers) & set(cand_name_strengths)):
                rejection_reasons.append(
                    f"Candidate specifies strength not verified in OCR text."
                )
                overall = min(overall, 55.0)

            if has_manufacturer_evidence and cand_mfr and mfr_score < 70.0:
                rejection_reasons.append(
                    f"Manufacturer mismatch: detected {detected_mfr} vs candidate {cand_mfr}."
                )
                overall = min(overall, 60.0)

        # CRITICAL HARD BRAND VALIDATION:
        # If a distinct brand was detected from the image / OCR (e.g. "P-650", "1-AL", "Dolo 650"):
        # A candidate whose brand does not match (brand_score < 55.0) CANNOT be
        # verified as the detected medicine!
        if brand_query and len(brand_query.strip()) >= 2 and brand_score < 55.0:
            composition_valid = False
            rejection_reasons.append(
                f"Brand mismatch: detected trade name '{brand_query}' does not match candidate '{cand_name}'"
            )

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


# ── Step 5 Multi-Evidence Matcher for Printed Packages ─────────────────────────

PHARMA_STOPWORDS = {
    'TABLETS', 'TABLET', 'CAPSULES', 'CAPSULE', 'CAPLETS', 'CAPLET', 'CREAM', 'SYRUP',
    'STRIP', 'STORE', 'PROTECT', 'EACH', 'CONTAINS', 'LIMITED', 'PHARMA', 'PHARMACEUTICAL',
    'PHARMACEUTICALS', 'LABORATORIES', 'LAB', 'LABS', 'BATCH', 'MFG', 'EXP', 'DATE', 'MRP',
    'PRICE', 'RS', 'INR', 'USE', 'FOR', 'ONLY', 'KEEP', 'OUT', 'REACH', 'CHILDREN', 'COOL',
    'DRY', 'PLACE', 'SCHEDULE', 'PRESCRIPTION', 'DRUG', 'CAUTION', 'WARNING', 'DOSAGE',
    'DIRECTION', 'DIRECTIONS', 'DIRECTED', 'PHYSICIAN', 'UNCOATED', 'FILM', 'COATED',
    'SUSPENSION', 'INJECTION', 'INJ', 'DROPS', 'SOLUTION', 'LOTION', 'GEL', 'OINTMENT',
    'MOUTH', 'WASH', 'TOPICAL', 'ORAL', 'EXTERNAL', 'PVT', 'LTD', 'INC', 'CORP', 'CO',
    'COMPANY', 'INDIA', 'IP', 'BP', 'USP', 'NF', 'MG', 'ML', 'MCG', 'GM', 'GRAM', 'GRAMS',
    'IU', 'PERCENT', 'TAPE', 'ADHESIVE', 'GAUZE', 'BANDAGE', 'COTTON', 'SURGICAL', 'STERILE',
    'NET', 'QTY', 'QUANTITY', 'PACK', 'PACKAGE', 'TREATMENT', 'DERMA', 'DERMATOLOGICAL',
    'SHODILATOR', 'MUCOLYTIC', 'EXPECTORANT', 'NATURAL', 'SUGAR', 'FREE', 'TASTE', 'WITH',
    'DYE', 'RELIEF', 'EXTRA', 'FAST', 'ACTION', 'PLUS', 'FORTE', 'CARE', 'HEALTHCARE',
    'FEVER', 'PAIN', 'ANTI', 'INFLAMMATORY', 'ANALGESIC', 'ANTIPYRETIC', 'ITCH', 'ACNE',
    'ALLERGY', 'COLD', 'COUGH', 'ORANGE', 'FLAVOUR', 'FLAVORED', 'BORG', 'LEI', 'CEIP',
    'LOLIG', 'TODUCE', 'TOCUCE', 'JULPHAR', 'JAMJOM', 'TOR', 'EASTED', 'SIN', 'REMO',
    'SVIAMOWING', 'IVET', 'KAR', 'YLY', 'WYONI', 'ARRIV', 'ARGICAL', 'AIR', 'VENTED',
    '5CMX5YD', 'NAN', 'TAPE', 'OINT', 'COMPOSITION'
}

COMMON_SALTS = {
    'PARACETAMOL', 'ACETAMINOPHEN', 'IBUPROFEN', 'TRETINOIN', 'ACETYLSALICYLIC', 'ASPIRIN',
    'AMOXICILLIN', 'AMOXYCILLIN', 'CLAVULANIC', 'CLAVULANATE', 'CETIRIZINE', 'LEVOCETIRIZINE',
    'AZITHROMYCIN', 'DICLOFENAC', 'ACECLOFENAC', 'PANTOPRAZOLE', 'OMEPRAZOLE', 'RABEPRAZOLE',
    'METFORMIN', 'GLIMEPIRIDE', 'ATORVASTATIN', 'ROSUVASTATIN', 'AMLODIPINE', 'TELMISARTAN',
    'SALBUTAMOL', 'ALBUTEROL', 'POVIDONE', 'IODINE', 'BETAMETHASONE', 'CELECOXIB', 'METRONIDAZOLE',
    'LACTULOSE', 'TERBINAFINE', 'DOMPERIDONE', 'FUSIDIC', 'GENTAMICIN', 'SITAGLIPTIN', 'ACYCLOVIR',
    'FEXOFENADINE'
}


def split_glued_tokens(text: str) -> str:
    """Split glued tokens at letter/digit boundaries (e.g. ACRETIN0.05% -> ACRETIN 0.05%)."""
    t = re.sub(r'([A-Za-z]+)(\d+(?:\.\d+)?%?)', r'\1 \2', text)
    t = re.sub(r'(\d+(?:\.\d+)?%?)([A-Za-z]+)', r'\1 \2', t)
    return t


_ENGLISH_WORDS = None

def get_english_words() -> set:
    """Load english_words.txt guard list."""
    global _ENGLISH_WORDS
    if _ENGLISH_WORDS is None:
        p = Path(__file__).resolve().parent / "data" / "english_words.txt"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                _ENGLISH_WORDS = {line.strip().lower() for line in f if line.strip()}
        else:
            _ENGLISH_WORDS = set()
    return _ENGLISH_WORDS


def assign_confidence_label(cand: Dict[str, Any]) -> str:
    """
    Derives display label ("Strong", "Fair", "Weak") from existing confidence flags,
    NOT from the raw internal score number.
    """
    is_exact = bool(cand.get("is_exact_brand"))
    corroborated = bool(cand.get("corroborated"))
    salt_matched = bool(cand.get("salt_matched"))
    strength_matched = bool(cand.get("strength_matched"))
    confident = bool(cand.get("confident"))

    if (is_exact and corroborated) or (salt_matched and strength_matched):
        return "Strong"
    if is_exact or corroborated or salt_matched or confident:
        return "Fair"
    return "Weak"


def match_printed_lines(
    lines: List[str],
    cutoff: Optional[float] = None,
    brand_index: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Multi-evidence matcher for printed packaging lines.
    1. Splits glued tokens (letter/digit boundaries).
    2. Tests word alone, hyphen-joined, adjacent 2-word joins.
    3. Filters pharmaceutical stop-list.
    4. Fuzzy matches against BrandWordIndex (~165k entries, rapidfuzz fuzz.ratio, cutoff 75).
    5. Corroborates with salt evidence (+35), strengths (+15/-10), manufacturer (+10), supporting lines (+3).
    6. Enforces compound-count guard.
    7. Returns top 3 candidates and 'not_sure' status if confidence is below threshold.
    """
    from django.conf import settings
    from .medicine_index import get_brand_word_index

    if cutoff is None:
        cutoff = getattr(settings, 'MATCH_CUTOFF', 75.0)

    # Use search cutoff ~70 so candidates are not prematurely dropped
    search_cutoff = min(float(cutoff), 70.0)

    if brand_index is None:
        brand_index = get_brand_word_index()

    full_text = "\n".join(lines).upper()
    pack_strengths = extract_strengths_from_text(full_text)
    detected_strengths = pack_strengths or set(re.findall(r'\b\d+(?:\.\d+)?%?\b', full_text))
    is_generic_salt = False
    generic_salt_display = ""

    english_words = get_english_words()

    # Guard A: Brand tokens must strictly come from prominent brand boxes
    # (never sentences, instructions, or boilerplate lines).
    candidate_tokens: Dict[str, Tuple[float, float, int]] = {}
    brand_eligible_tokens: Set[str] = set()

    def _is_safe_brand_token(tok: str) -> bool:
        if not tok or len(tok) < 3 or tok.isdigit():
            return False
        tok_l = tok.lower()
        if tok in PHARMA_STOPWORDS:
            return False
        if len(tok) <= 5:
            if (tok_l in english_words
                or tok_l in COMMON_FRAGMENTS
                or tok in PHARMA_STOPWORDS
                or tok in GRAMMAR_WORDS):
                return False
        if any(re.search(pat, tok) for pat in BOILERPLATE_BOX_PATTERNS):
            return False
        return True

    for box_idx, line in enumerate(lines):
        line_clean = line.strip().upper()
        if not line_clean:
            continue
        # Guard A: ignore boxes that match BOILERPLATE_BOX_PATTERNS before brand matching
        if any(re.search(pat, line_clean) for pat in BOILERPLATE_BOX_PATTERNS):
            continue
        # Guard A: a brand counts only if it is a whole text box or a whole word of a short box
        if not is_brand_box_candidate(line):
            continue

        prominence = 20.0 if box_idx == 0 else (12.0 if box_idx == 1 else (6.0 if box_idx == 2 else 0.0))
        prominence += 10.0

        cleaned = split_glued_tokens(line.upper())
        words = [re.sub(r'[^A-Z0-9]', '', w) for w in re.split(r'[\s\-_]+', cleaned) if w.strip()]

        # 1. Word alone
        for w in words:
            if _is_safe_brand_token(w):
                cur_sc, cur_prom, cur_cnt = candidate_tokens.get(w, (0.0, 0.0, 0))
                candidate_tokens[w] = (cur_sc, max(cur_prom, prominence), cur_cnt + 1)
                brand_eligible_tokens.add(w)

        # 2. Hyphen joined
        hyphen_j = re.sub(r'[^A-Z0-9]', '', line.upper())
        if _is_safe_brand_token(hyphen_j):
            cur_sc, cur_prom, cur_cnt = candidate_tokens.get(hyphen_j, (0.0, 0.0, 0))
            candidate_tokens[hyphen_j] = (cur_sc, max(cur_prom, prominence), cur_cnt + 1)
            brand_eligible_tokens.add(hyphen_j)

        # 3. Two-word joins
        for i in range(len(words) - 1):
            two_j = words[i] + words[i + 1]
            if _is_safe_brand_token(two_j):
                cur_sc, cur_prom, cur_cnt = candidate_tokens.get(two_j, (0.0, 0.0, 0))
                candidate_tokens[two_j] = (cur_sc, max(cur_prom, prominence), cur_cnt + 1)
                brand_eligible_tokens.add(two_j)

    # A) BRAND-TOKEN GUARD:
    exact_brand_words = set()
    for tok in brand_eligible_tokens:
        tok_l = tok.lower()
        if not _is_safe_brand_token(tok) or tok in COMMON_SALTS:
            continue
        if tok in brand_index.brand_to_ids:
            exact_brand_words.add(tok)
        elif len(tok) >= 4:
            hit = process.extractOne(tok, brand_index.unique_brand_words, scorer=fuzz.ratio, score_cutoff=88.0)
            if hit:
                h_word = hit[0]
                if _is_safe_brand_token(h_word) and h_word not in COMMON_SALTS:
                    exact_brand_words.add(h_word)

    brand_scores: Dict[str, Tuple[float, int]] = {}
    for tok, (_, prom, count) in candidate_tokens.items():
        pop_bonus = min(10.0, math.log2(brand_index.brand_counts.get(tok, 1) + 1) * 2.0) if hasattr(brand_index, 'brand_counts') else 0.0
        if tok in exact_brand_words:
            score = 150.0 + prom + pop_bonus
            brand_scores[tok] = (score, count)
        elif tok in brand_index.brand_to_ids:
            score = 100.0 + prom + pop_bonus
            if tok in COMMON_SALTS:
                score -= 15.0
            if tok not in brand_scores or score > brand_scores[tok][0]:
                brand_scores[tok] = (score, count)
        elif len(tok) >= 4:
            f_matches = process.extract(tok, brand_index.unique_brand_words, scorer=fuzz.ratio, limit=3, score_cutoff=search_cutoff)
            for m_b, score, _ in f_matches:
                if m_b in exact_brand_words:
                    adj_score = 150.0 + prom
                else:
                    let_bonus = 0.0
                    if len(m_b) >= 3 and len(tok) >= 3 and m_b[:3] == tok[:3]:
                        let_bonus = 15.0
                    elif len(m_b) >= 2 and len(tok) >= 2 and m_b[:2] == tok[:2]:
                        let_bonus = 10.0
                    elif len(m_b) >= 1 and len(tok) >= 1 and m_b[0] == tok[0]:
                        let_bonus = 5.0
                    m_pop = min(10.0, math.log2(brand_index.brand_counts.get(m_b, 1) + 1) * 2.0) if hasattr(brand_index, 'brand_counts') else 0.0
                    adj_score = score + prom + let_bonus + m_pop
                    if m_b in COMMON_SALTS:
                        adj_score -= 15.0
                if m_b not in brand_scores or adj_score > brand_scores[m_b][0]:
                    brand_scores[m_b] = (adj_score, count)

    for eb in exact_brand_words:
        if eb not in brand_scores:
            brand_scores[eb] = (150.0 + 12.0, 1)

    # Salt extraction across all OCR lines (including composition & dosage lines)
    pack_salts = extract_active_salt_set(full_text)

    # ── STEP 2: BRAND-LESS PATH (levocetirizine strip, and generic packs) ──
    # When no brand passes the brand guard:
    # a) Do NOT choose a brand by fuzzy score, popularity, or salt-brand links.
    #    Never return a brand name that is not supported by the pack text.
    # b) Agreeing evidence:
    #    - salt set: must equal what was read on the pack.
    #    - strength: must equal what was read (if read).
    #    - dosage form: tablet.
    #    - manufacturer: if read on the pack ("FDC Limited"), filter the DB for
    #      medicines manufactured by that company (>= 90 ratio, ignoring Ltd/Limited/Pvt).
    # c) Filter the DB:
    #    - If exactly 1 row is left: that is your result. Level = Likely.
    #      Message: "Matched by salt, strength and manufacturer; the brand name was not readable on the pack."
    #    - If several rows or 0 rows are left: return ONE generic card:
    #      "Levocetirizine 5 mg tablets, manufactured by FDC Limited (brand name not readable on the pack)"
    #      Level = Likely. No brand name. No price.
    #    - If manufacturer was not read: "Levocetirizine 5 mg tablets (brand name not readable on the pack)".
    # d) Small brand tag second reading:
    #    - If any recognized token normalized matches a candidate brand (e.g. 1-AL / L-AL) exactly,
    #      accept the brand. Only accept exact normalized match.
    if not exact_brand_words and pack_salts:
        from .extractor import extract_manufacturer

        # 1. Extract manufacturer from pack lines
        pack_mfr = None
        for l in lines:
            l_clean = l.strip().upper()
            if re.search(r'\b(?:SCHEDULE|PHYSICIAN|PRESCRIPTION|DRUG|TEMPERATURE|PROTECT|DIRECTED)\b', l_clean):
                continue
            m = re.search(r'\b([A-Z0-9\s\.\-]{2,40}\b(?:LTD|LIMITED|PVT|PHARMA|LABORATORIES|LABS|PHARMACEUTICALS))\b', l_clean)
            if m:
                cand_mfr_str = m.group(1).strip()
                cand_mfr_str = re.sub(r'^O(?=[A-Z]{2,})', '', cand_mfr_str)
                cand_mfr_str = re.sub(r'\s+', ' ', cand_mfr_str)
                pack_mfr = cand_mfr_str.title()
                break

        if not pack_mfr:
            extracted_mfr = extract_manufacturer(full_text)
            if extracted_mfr:
                pack_mfr = re.sub(r'^O(?=[A-Z]{2,})', '', extracted_mfr.strip()).title()

        def _clean_mfr_name(m: Optional[str]) -> str:
            if not m:
                return ""
            s = re.sub(r'\b(ltd|limited|pvt|private|pharma|pharmaceuticals|laboratories|labs|inc|corp)\b', '', m.lower())
            return re.sub(r'[^a-z0-9]', '', s).strip()

        # Query candidates for this salt set & manufacturer
        matching_rows = []
        pack_mfr_clean = _clean_mfr_name(pack_mfr)
        if pack_mfr_clean:
            try:
                db_file = getattr(brand_index, 'db_path', None) or Path(settings.MEDICINES_DB_PATH)
                import sqlite3
                conn = sqlite3.connect(str(db_file))
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()

                # Filter in SQL by manufacturer to retrieve only matching rows
                mfr_prefix = pack_mfr_clean[:4]
                cur.execute(
                    "SELECT id, name, salt, manufacturer, price FROM medicines WHERE LOWER(manufacturer) LIKE ? LIMIT 200",
                    (f"%{mfr_prefix}%",)
                )
                raw_mfr_rows = [dict(r) for r in cur.fetchall()]
                conn.close()

                for r in raw_mfr_rows:
                    r_salts = extract_active_salt_set(r.get("salt") or "")
                    if r_salts != pack_salts:
                        continue

                    r_strengths = extract_strengths_from_text(f"{r.get('name')} {r.get('salt')}")
                    if pack_strengths and not (r_strengths & pack_strengths):
                        continue

                    r_mfr = r.get("manufacturer") or ""
                    r_mfr_clean = _clean_mfr_name(r_mfr)
                    if not r_mfr_clean:
                        continue
                    sim = fuzz.ratio(pack_mfr_clean, r_mfr_clean)
                    if sim >= 90 or pack_mfr_clean == r_mfr_clean:
                        matching_rows.append(r)
            except Exception as exc:
                logger.warning("Error filtering DB for brandless path: %s", exc)

        # d) Check small brand tag secondary reading
        matched_small_tag = None
        all_norm_tokens = {re.sub(r'[^A-Z0-9]', '', l.upper()) for l in lines}
        for l in lines:
            for w in re.split(r'[\s\-_]+', l.upper()):
                all_norm_tokens.add(re.sub(r'[^A-Z0-9]', '', w))

        if matching_rows:
            for r in matching_rows:
                cand_brand = re.split(r'[\s\-_]+', r["name"])[0]
                norm_brand = re.sub(r'[^A-Z0-9]', '', cand_brand.upper())
                alt_norm_brands = {norm_brand}
                if norm_brand.startswith('1'):
                    alt_norm_brands.add('L' + norm_brand[1:])
                    alt_norm_brands.add('I' + norm_brand[1:])
                elif norm_brand.startswith('L') or norm_brand.startswith('I'):
                    alt_norm_brands.add('1' + norm_brand[1:])

                if alt_norm_brands & all_norm_tokens:
                    matched_small_tag = r
                    break

        salt_display_parts = [s.title() for s in sorted(pack_salts)]
        salt_title = " + ".join(salt_display_parts)
        p_disp = ", ".join([f"{st} mg" if not st.endswith("mg") and not st.endswith("%") else st for st in sorted(pack_strengths)]) if pack_strengths else ""

        # Case 1: Small brand tag matched an exact candidate brand
        if matched_small_tag:
            exact_cand_brand = re.split(r'[\s\-_]+', matched_small_tag["name"])[0]
            exact_brand_words.add(exact_cand_brand.upper())
            brand_scores[exact_cand_brand.upper()] = (150.0 + 12.0, 1)

        # Case 2: Exactly 1 row left by salt, strength, and manufacturer
        elif len(matching_rows) == 1:
            m_row = matching_rows[0]
            first_b = re.split(r'[\s\-_]+', m_row["name"])[0]
            med_dict = {
                "id": m_row["id"],
                "name": m_row["name"],
                "detected_name": m_row["name"],
                "brand": first_b,
                "salt": m_row.get("salt") or salt_title,
                "strength": p_disp,
                "manufacturer": m_row.get("manufacturer") or pack_mfr or "Registered Manufacturer",
                "price": m_row.get("price"),
                "dosage_form": "Tablet",
                "score": 85.0,
                "intra_score": 85.0,
                "is_exact_brand": False,
                "match_type": "brandless_single",
                "confident": True,
                "corroborated": True,
                "salt_matched": True,
                "salt_ok": True,
                "strength_matched": True,
                "strength_mismatch": False,
            }
            breakdown = {
                "brand": 40.0, "ingredients": 95.0, "strength": 90.0 if pack_strengths else 70.0,
                "manufacturer": 90.0 if pack_mfr else 50.0, "dosage_form": 80.0, "overall": 80.0
            }
            msg = "Matched by salt, strength and manufacturer; the brand name was not readable on the pack."
            return {
                "success": False,
                "status": "likely",
                "result_level": "likely",
                "match_type": "brandless_single",
                "medicine": med_dict,
                "candidates": [med_dict],
                "medicines": [med_dict],
                "top_candidates": [med_dict],
                "score_margin": 0.0,
                "confidence_breakdown": breakdown,
                "raw_ocr": full_text,
                "n_matching_tokens": len(pack_salts),
                "is_generic_salt": False,
                "generic_salt_display": m_row["name"],
                "strength_mismatch_msg": "",
                "pack_strengths": list(pack_strengths),
                "pack_salts": list(pack_salts),
                "message": msg
            }

        # Case 3: 0 rows or several rows -> ONE generic card
        else:
            # Bug 2: Check if a brand-like token was read on the pack (e.g. P-650)
            unindexed_brand_token = None
            for l_box in lines:
                l_clean = l_box.strip().upper()
                if not l_clean or len(l_clean) < 2:
                    continue
                if any(re.search(pat, l_clean) for pat in BOILERPLATE_BOX_PATTERNS):
                    continue
                if re.search(r'\b(?:DOSE|DOSES|DAILY|MAXIMUM|MAX|EVERY|HOURS?|PER\s*DAY)\b', l_clean):
                    continue
                # Match whole box with letters and digits (e.g. P-650) or a prominent safe brand token
                if re.match(r'^[A-Z0-9]+(?:[- ][A-Z0-9]+)?$', l_clean):
                    if any(c.isalpha() for c in l_clean):
                        b_words = re.split(r'[\s\-_]+', l_clean)
                        if not any(w.lower() in pack_salts or w.lower() in COMMON_SALTS for w in b_words):
                            if not any(w in PHARMA_STOPWORDS or w in GRAMMAR_WORDS or w in {'LIMITED', 'LTD', 'PVT', 'PHARMA', 'TABLETS', 'TABLET', 'CAPSULES', 'CAPSULE', 'IP', 'BP', 'USP', 'COMPOSITION'} for w in b_words):
                                if (any(c.isdigit() for c in l_clean) and any(c.isalpha() for c in l_clean)) or (len(b_words) == 1 and len(l_clean) >= 4 and _is_safe_brand_token(l_clean)):
                                    unindexed_brand_token = l_box.strip()
                                    break

            if unindexed_brand_token:
                brand_tag_note = f"(brand {unindexed_brand_token} not in database)"
                msg = f"Brand {unindexed_brand_token} was read on the pack but is not in the database. Showing the generic medicine."
            else:
                brand_tag_note = "(brand name not readable on the pack)"
                msg = "Active ingredient and strength matched from packaging. Brand name not readable on the pack."

            if pack_mfr:
                if p_disp:
                    generic_salt_display = f"{salt_title} {p_disp} tablets, manufactured by {pack_mfr} {brand_tag_note}"
                else:
                    generic_salt_display = f"{salt_title} tablets, manufactured by {pack_mfr} {brand_tag_note}"
            else:
                if p_disp:
                    generic_salt_display = f"{salt_title} {p_disp} tablets {brand_tag_note}"
                else:
                    generic_salt_display = f"{salt_title} tablets {brand_tag_note}"

            generic_med = {
                "id": None,
                "name": generic_salt_display,
                "detected_name": generic_salt_display,
                "brand": "",
                "salt": salt_title,
                "strength": p_disp,
                "manufacturer": pack_mfr or "Generic Packaging (Read from Pack)",
                "price": None,
                "dosage_form": "Tablet",
                "score": 85.0,
                "intra_score": 85.0,
                "is_exact_brand": False,
                "match_type": "generic_salt",
                "confident": True,
                "corroborated": True,
                "salt_matched": True,
                "salt_ok": True,
                "strength_matched": True,
                "strength_mismatch": False,
            }
            breakdown = {
                "brand": 0.0, "ingredients": 95.0, "strength": 90.0 if pack_strengths else 70.0,
                "manufacturer": 85.0 if pack_mfr else 50.0, "dosage_form": 80.0, "overall": 85.0
            }
            return {
                "success": False,
                "status": "likely",
                "result_level": "likely",
                "match_type": "generic_salt",
                "medicine": generic_med,
                "candidates": [generic_med],
                "medicines": [generic_med],
                "top_candidates": [generic_med],
                "score_margin": 0.0,
                "confidence_breakdown": breakdown,
                "raw_ocr": full_text,
                "n_matching_tokens": len(pack_salts),
                "is_generic_salt": True,
                "generic_salt_display": generic_salt_display,
                "strength_mismatch_msg": "",
                "pack_strengths": list(pack_strengths),
                "pack_salts": list(pack_salts),
                "message": msg
            }

    # Count read tokens matching brand/salt index with similarity >= 90
    n_matching_tokens = 0
    for tok in candidate_tokens:
        if tok in brand_index.brand_to_ids or (hasattr(brand_index, 'salt_to_brand_words') and tok.lower() in brand_index.salt_to_brand_words):
            n_matching_tokens += 1
        elif len(tok) >= 4:
            b_hit = process.extractOne(tok, brand_index.unique_brand_words, scorer=fuzz.ratio, score_cutoff=90.0)
            if b_hit:
                n_matching_tokens += 1
            elif hasattr(brand_index, 'unique_salt_words') and brand_index.unique_salt_words:
                s_hit = process.extractOne(tok.lower(), brand_index.unique_salt_words, scorer=fuzz.ratio, score_cutoff=90.0)
                if s_hit:
                    n_matching_tokens += 1

    if not brand_scores:
        return {
            "success": False,
            "status": "unverified",
            "result_level": "not_found",
            "medicine": None,
            "candidates": [],
            "medicines": [],
            "top_candidates": [],
            "confidence_breakdown": _zero_breakdown(),
            "raw_ocr": full_text,
            "n_matching_tokens": n_matching_tokens,
            "message": "No matching brand words found in local database."
        }

    # Gather candidate medicine rows from SQLite
    cand_ids = []
    brand_by_id = {}

    # 1. ALWAYS retrieve exact brands first
    for eb in sorted(exact_brand_words):
        b_score, b_count = brand_scores.get(eb, (160.0, 1))
        for mid in brand_index.brand_to_ids.get(eb, [])[:15]:
            cand_ids.append(mid)
            brand_by_id[mid] = (eb, b_score, b_count, True)

    # 2. Retrieve remaining top brand words
    for b_word, (b_score, b_count) in sorted(brand_scores.items(), key=lambda x: x[1][0], reverse=True)[:25]:
        if b_word not in exact_brand_words:
            for mid in brand_index.brand_to_ids.get(b_word, [])[:15]:
                cand_ids.append(mid)
                brand_by_id[mid] = (b_word, b_score, b_count, False)

    rows_db = brand_index.get_medicines_by_ids(cand_ids)

    scored_candidates = []
    for row in rows_db:
        mid = row["id"]
        name = row["name"]
        salt = row.get("salt") or ""
        mfr = row.get("manufacturer") or ""
        price = row.get("price")

        b_word, b_score, b_count, is_exact = brand_by_id.get(mid, ("", 0.0, 1, False))
        cand_name = (name or "").upper()
        cand_salt = salt.upper()
        cand_mfr = mfr.upper()

        score = b_score

        cand_salts = extract_active_salt_set(salt)
        cand_strengths = extract_strengths_from_text(f"{cand_name} {cand_salt}")

        # 1. Salt comparison
        salt_matched = False
        salt_ok = False
        if pack_salts:
            if cand_salts == pack_salts:
                salt_matched = True
                salt_ok = True
                score += 35.0
            elif cand_salts.issubset(pack_salts):
                salt_matched = True
                score += 15.0
            else:
                # Combination product with extra salts not read on the pack (e.g. Sted-ML extra montelukast)
                score -= 60.0
        else:
            salt_words = [re.sub(r'[^A-Z0-9]', '', w) for w in re.split(r'[\s,()/+]+', cand_salt) if len(w) >= 4]
            if any(sw in full_text and sw not in PHARMA_STOPWORDS for sw in salt_words):
                salt_matched = True
                salt_ok = True
                score += 35.0

        # 2. Strength comparison
        strength_matched = False
        strength_mismatch = False
        if pack_strengths:
            if cand_strengths & pack_strengths:
                strength_matched = True
                score += 25.0
            elif cand_strengths:
                # Candidate has a strength, but it does NOT match the pack strength
                strength_mismatch = True
                score -= 25.0
        elif detected_strengths and cand_strengths:
            # Fallback detected strengths
            if any(st in cand_strengths for st in detected_strengths if len(st) >= 2 or st in {'5', '2'}):
                strength_matched = True
                score += 15.0
            else:
                score -= 10.0

        # 3. Manufacturer boost (+10)
        if cand_mfr:
            mfr_words = [w for w in re.split(r'[\s,.]+', cand_mfr) if len(w) >= 4 and w not in PHARMA_STOPWORDS]
            if any(mw in full_text for mw in mfr_words):
                score += 10.0

        # 4. Multi-line support (+3 per extra line)
        if b_count > 1:
            score += min(9.0, (b_count - 1) * 3.0)

        first_brand = re.split(r'[\s\-_]+', name.strip())[0]
        is_in_english_words = (b_word.lower() in english_words) or (first_brand.lower() in english_words)
        if is_exact and is_in_english_words:
            if not (salt_matched or strength_matched):
                is_exact = False
                score -= 50.0  # Remove exact-brand priority boost

        # Scoped intra-brand compound guard
        intra_pref = 0.0
        if '+' in cand_salt and '+' not in full_text and len(pack_salts) <= 1:
            intra_pref = -15.0

        corroborated = bool(salt_matched or strength_matched)
        confident = bool(score >= cutoff)

        b_norm_cand = re.sub(r'[^A-Z0-9]', '', first_brand.upper())
        cand_match_type = "ingredient" if (b_word in COMMON_SALTS or b_norm_cand in COMMON_SALTS) else "brand"

        cand_dict = {
            "id": mid,
            "name": name,
            "brand": first_brand,
            "salt": salt,
            "manufacturer": mfr or "Registered Manufacturer",
            "price": price,
            "score": round(score, 1),
            "intra_score": round(score + intra_pref, 1),
            "is_exact_brand": is_exact,
            "match_type": cand_match_type,
            "confident": confident,
            "corroborated": corroborated,
            "salt_matched": salt_matched,
            "salt_ok": salt_ok,
            "strength_matched": strength_matched,
            "strength_mismatch": strength_mismatch,
            "dosage_form": "Tablet"
        }
        cand_dict["confidence_label"] = assign_confidence_label(cand_dict)
        scored_candidates.append(cand_dict)

    # Deduplicate candidates by brand name, using intra_score to select the best row for each brand
    by_brand = {}
    for cand in scored_candidates:
        b_norm = re.sub(r'[^A-Z0-9]', '', cand["brand"].upper())
        if b_norm not in by_brand or cand["intra_score"] > by_brand[b_norm]["intra_score"]:
            by_brand[b_norm] = cand

    dedup_candidates = list(by_brand.values())

    # Sort candidates ensuring exact brand matches are at the top
    exact_cands = [c for c in dedup_candidates if c.get("is_exact_brand")]
    non_exact_cands = [c for c in dedup_candidates if not c.get("is_exact_brand")]
    exact_cands.sort(key=lambda x: x["score"], reverse=True)
    non_exact_cands.sort(key=lambda x: x["score"], reverse=True)
    sorted_candidates = (exact_cands + non_exact_cands)[:5]

    for i in range(1, len(sorted_candidates)):
        if sorted_candidates[i]["score"] > sorted_candidates[i - 1]["score"]:
            sorted_candidates[i]["score"] = sorted_candidates[i - 1]["score"]

    if not sorted_candidates:
        return {
            "success": False,
            "status": "unverified",
            "result_level": "not_found",
            "match_type": "brand",
            "medicine": None,
            "candidates": [],
            "medicines": [],
            "top_candidates": [],
            "confidence_breakdown": _zero_breakdown(),
            "raw_ocr": full_text,
            "message": "Unable to find candidate medicines in local database."
        }

    top_cand = sorted_candidates[0]
    sec_cand = sorted_candidates[1] if len(sorted_candidates) > 1 else None
    score_margin = round(top_cand["score"] - sec_cand["score"], 1) if sec_cand else top_cand["score"]
    top_cand["score_margin"] = score_margin

    match_type = top_cand.get("match_type", "brand")

    # B) NEW VERIFIED RULE (replace is_rule4). Verified requires ALL of:
    # 1. exact brand passes the guard in A;
    # 2. the candidate salt set EQUALS the salt set read on the pack. Use a set
    #    comparison, not any(). A combination with extra salts fails;
    # 3. the strength on the pack equals the candidate strength. If the pack has
    #    no readable strength, strength is not required, but the result is then
    #    capped at Likely when the brand has more than one strength in the DB.
    # Otherwise the level is Likely, or Not found if nothing is safe.

    top_cand_brand = top_cand.get("brand", "").upper()
    top_cand_salts = extract_active_salt_set(top_cand.get("salt"))
    top_cand_strengths = extract_strengths_from_text(f"{top_cand.get('name')} {top_cand.get('salt')}")

    # Condition 1: Exact brand passes guard in A
    cond1_brand = bool(
        top_cand.get("is_exact_brand")
        and not is_generic_salt
        and top_cand.get("match_type") != "ingredient"
    )

    # Condition 2: Candidate salt set EQUALS salt set read on the pack
    if pack_salts:
        cond2_salt = bool(top_cand_salts == pack_salts)
    else:
        cond2_salt = bool(top_cand.get("salt_matched"))

    # Condition 3: Strength on pack equals candidate strength
    if pack_strengths:
        cond3_strength = bool(top_cand_strengths & pack_strengths)
    else:
        # If the pack has no readable strength, strength is not required,
        # but the result is then capped at Likely when the brand has more than one strength in the DB.
        brand_ids = brand_index.brand_to_ids.get(top_cand_brand, [])
        brand_meds = brand_index.get_medicines_by_ids(brand_ids[:10])
        distinct_strengths = set()
        for bm in brand_meds:
            s_set = extract_strengths_from_text(f"{bm.get('name')} {bm.get('salt')}")
            distinct_strengths.update(s_set)
        if len(distinct_strengths) > 1:
            cond3_strength = False  # Capped at Likely
        else:
            cond3_strength = True

    is_verified = bool(cond1_brand and cond2_salt and cond3_strength)
    status = "verified" if is_verified else "likely"
    result_level = "verified" if is_verified else "likely"

    strength_mismatch_msg = ""
    if top_cand.get("strength_mismatch"):
        p_disp = ", ".join(sorted(pack_strengths)) if pack_strengths else "unknown"
        c_disp = ", ".join(sorted(top_cand_strengths)) if top_cand_strengths else "unknown"
        strength_mismatch_msg = (
            f"Brand matches, but the strength on the pack ({p_disp}) is not in the database. "
            f"Closest listed: {c_disp}."
        )

    breakdown = {
        "brand": min(100.0, max(0.0, top_cand["score"])),
        "ingredients": 95.0 if top_cand.get("salt_matched") else 50.0,
        "strength": 85.0 if top_cand.get("strength_matched") else 50.0,
        "manufacturer": 70.0,
        "dosage_form": 80.0,
        "overall": min(100.0, max(0.0, top_cand["score"]))
    }

    return {
        "success": is_verified,
        "status": status,
        "result_level": result_level,
        "match_type": match_type,
        "medicine": top_cand,
        "candidates": sorted_candidates[:3],
        "medicines": sorted_candidates[:3],
        "top_candidates": sorted_candidates[:5],
        "score_margin": score_margin,
        "confidence_breakdown": breakdown,
        "raw_ocr": full_text,
        "n_matching_tokens": n_matching_tokens,
        "is_generic_salt": is_generic_salt,
        "generic_salt_display": generic_salt_display,
        "strength_mismatch_msg": strength_mismatch_msg,
        "pack_strengths": list(pack_strengths),
        "pack_salts": list(pack_salts),
        "message": (
            strength_mismatch_msg if strength_mismatch_msg else (
                "Matches the brand and the ingredient printed on the pack."
                if is_verified
                else (
                    "Closest match by ingredient"
                    if match_type == "ingredient"
                    else "Top candidate based on packaging text — please confirm brand and strength."
                )
            )
        )
    }

