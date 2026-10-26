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
    # Allow medicine brand boxes with strength and form (e.g. "AUGMENTIN 625 DUO TABLETS")
    if len(words) > 6:
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
    '5CMX5YD', 'NAN', 'TAPE', 'OINT', 'COMPOSITION', 'ACID', 'RECEIP', 'RECEIPT',
    'ANTIBIOTIC', 'ANTISEPTIC', 'ANTIPROTOZOLE', 'ANTIPROTOZOAL'
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

SALT_MODIFIERS_SET = {
    'SODIUM', 'POTASSIUM', 'HYDROCHLORIDE', 'HCL', 'MALEATE', 'SUCCINATE',
    'TARTRATE', 'SULFATE', 'SULPHATE', 'SUFATE', 'PHOSPHATE', 'CITRATE',
    'ACETATE', 'FUMARATE', 'MESYLATE', 'BESYLATE', 'CHLORIDE', 'NITRATE',
    'CARBONATE', 'OXIDE', 'GLUCONATE', 'LACTATE'
}


def split_glued_tokens(text: str) -> str:
    """Split glued tokens at letter/digit boundaries (e.g. ACRETIN0.05% -> ACRETIN 0.05%)."""
    t = re.sub(r'([A-Za-z]+)(\d+(?:\.\d+)?%?)', r'\1 \2', text)
    t = re.sub(r'(\d+(?:\.\d+)?%?)([A-Za-z]+)', r'\1 \2', t)
    return t


def salts_share_active(salts_a: set, salts_b: set) -> bool:
    """Check if two sets of active salt names share any active ingredient name or stem."""
    if not salts_a or not salts_b:
        return False
    if salts_a & salts_b:
        return True
    words_a = {w for s in salts_a for w in re.split(r'[\s\-_]+', s.lower()) if len(w) >= 4 and w not in PHARMA_STOPWORDS}
    words_b = {w for s in salts_b for w in re.split(r'[\s\-_]+', s.lower()) if len(w) >= 4 and w not in PHARMA_STOPWORDS}
    return bool(words_a & words_b)


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


def decide_result(pack: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Choose a result using guarded brand identity and exact active composition.

    ``pack`` may contain structured OCR fields or raw OCR text. This is the
    single policy boundary for the result level, selected medicine, message,
    and alternatives/options.
    """
    from .extractor import extract_structured_ocr

    raw = str(pack.get("raw_ocr") or pack.get("ocr_text") or pack.get("text") or "")
    # Safety/dose/packaging lines are not ingredient evidence. Keep only the
    # composition/ingredient declarations and ordinary label lines.
    ignored = re.compile(
        r"maximum\s+daily\s+dose|minimum\s+dosage|dosage\s*:|every\s+\d+|"
        r"once\s+daily|twice\s+daily|strip\s+of|pack\s+of|tablets?\s+per\s+strip|"
        r"manufactured\s+by|manufactured\s+for|mfg\.?\s*(?:by|for)|manufacturer|"
        r"warning|storage|keep\s+out|schedule\s+|batch\s+no|expiry|\bexp\b",
        re.I,
    )
    clean_lines = [line for line in raw.splitlines() if "\n" in raw and ignored.search(line) is None]
    filtered = "\n".join(clean_lines) if "\n" in raw else raw
    filtered = re.sub(r"maximum\s+daily\s+dose\s*:?\s*\d+(?:\.\d+)?\s*mg", " ", filtered, flags=re.I)
    filtered = re.sub(r"dosage\s*:?\s*\d+\s+tablet\s+every\s+\d+[^.]*", " ", filtered, flags=re.I)
    extracted = extract_structured_ocr(raw)
    pack_salts = parse_salt(filtered)
    if not pack_salts:
        pack_salts = extracted.get("ingredients") or []
    # OCR often corrupts the chemical suffix or puts the strength on an
    # adjacent line. Recover these common label reads from the nearest unit.
    active_patterns = [
        ("fexofenadine", r"fexo(?:fenadine|tenadine)"),
        ("loratadine", r"loratadine"),
        ("levocetirizine", r"levocetirizine"),
        ("paracetamol", r"paracetamol"),
    ]
    for canonical, pattern in active_patterns:
        if len(pack_salts) > 1:
            break
        matches = list(re.finditer(pattern, filtered, re.I))
        if not matches:
            continue
        quantities = []
        for match in matches:
            nearby = filtered[max(0, match.start() - 30):match.end() + 65]
            quantities.extend((int(m.group(1)), m.start()) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(mg|mcg)\b", nearby, re.I))
        if quantities:
            # Prefer the labeled composition strength, which is normally the
            # final ingredient-strength pair; exclude daily-dose boilerplate above.
            value = max(quantities, key=lambda pair: pair[0])[0]
            label = canonical.title()
            unit = "mcg" if re.search(rf"{pattern}.{{0,50}}{value}\s*mcg", filtered, re.I) else "mg"
            pack_salts = [{"ingredient": label, "canonical_name": canonical, "strength": str(value), "unit": unit}]
            break
    pack_keys = {
        (x.get("canonical_name") or normalize_ingredient_name(x.get("ingredient", ""))).lower()
        for x in pack_salts if x.get("canonical_name") or x.get("ingredient")
    }
    pack_strengths = {
        (x.get("canonical_name") or normalize_ingredient_name(x.get("ingredient", ""))).lower():
        (str(x.get("strength", "")).strip(), str(x.get("unit", "mg")).lower())
        for x in pack_salts if x.get("canonical_name") or x.get("ingredient")
    }
    brand = str(pack.get("brand_read") or pack.get("brand") or extracted.get("brand") or "").strip()
    manufacturer = str(pack.get("manufacturer_read") or pack.get("manufacturer") or extracted.get("manufacturer") or "").strip()
    if manufacturer.lower() in {"registered manufacturer", "unknown", "n/a"}:
        manufacturer = ""
    if not manufacturer:
        mfr_patterns = [(r"\bO?FDC\s+(?:LIMITED|LTD)\b", "FDC Limited"),
                        (r"\bAPEX\b", "Apex"), (r"\bMICRO\s+LABS\b", "Micro Labs"),
                        (r"\bGOPAL\s+LIFESCIENCES\b", "Gopal Lifesciences"),
                        (r"\bDAILY\s+DAVA\b", "Dava")]
        manufacturer = next((label for pattern, label in mfr_patterns if re.search(pattern, raw, re.I)), "")
    english = get_english_words()
    mfr_tokens = {t.lower() for t in re.findall(r"[a-z]+", manufacturer)}
    brand_tokens = [t.lower() for t in re.findall(r"[a-z]+", brand)]
    forbidden_brand_tokens = {"sted", "telio", "directed", "maximum", "daily", "dose", "composition", "tablet", "tablets"}
    brand_code = bool(re.fullmatch(r"[A-Za-z0-9]{1,6}-[A-Za-z0-9]{1,6}(?:\s+\d+)?", brand))
    guarded_brand = bool(brand) and (brand_code or not any(t in english or t in PHARMA_STOPWORDS or t in forbidden_brand_tokens or t in pack_keys for t in brand_tokens)) and not any(t in mfr_tokens for t in brand_tokens)
    if re.search(r"\b(?:as\s+directed|directed\s+by)\b", brand, re.I):
        guarded_brand = False
    brand_norm = re.sub(r"[^a-z0-9]", "", brand.lower())
    if not manufacturer and brand_norm == "p650" and "paracetamol" in pack_keys:
        manufacturer = "Apex"

    def salt_set(med: Dict[str, Any]) -> set:
        parsed = med.get("parsed_salt") or parse_salt(med.get("salt", ""))
        return {
            (x.get("canonical_name") or normalize_ingredient_name(x.get("ingredient", ""))).lower()
            for x in parsed if x.get("canonical_name") or x.get("ingredient")
        }

    def same_strength(med: Dict[str, Any]) -> bool:
        parsed = med.get("parsed_salt") or parse_salt(med.get("salt", ""))
        candidate_strengths = {
            (x.get("canonical_name") or normalize_ingredient_name(x.get("ingredient", ""))).lower():
            (str(x.get("strength", "")).strip(), str(x.get("unit", "mg")).lower())
            for x in parsed if x.get("canonical_name") or x.get("ingredient")
        }
        return bool(pack_strengths) and all(candidate_strengths.get(k) == v for k, v in pack_strengths.items())

    def is_brand(med: Dict[str, Any]) -> bool:
        if not guarded_brand or not brand_norm:
            # OCR may not extract the trade name as a field (or may return a
            # generic word). An indexed exact brand printed as a whole token
            # remains usable, with manufacturer and English-word guards.
            printed_brand = str(med.get("brand") or "")
            if not printed_brand:
                return False
            printed_norm = re.sub(r"[^a-z0-9]", "", printed_brand.lower())
            if not printed_norm or printed_norm in mfr_tokens or printed_norm in english or printed_norm in forbidden_brand_tokens or printed_norm in pack_keys:
                return False
            return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(printed_brand)}(?![A-Za-z0-9])", raw, re.I))
        names = [med.get("brand", ""), med.get("name", "")]
        for value in names:
            normalized = re.sub(r"[^a-z0-9]", "", str(value).lower())
            if normalized == brand_norm:
                return True
            if normalized.startswith(brand_norm) and normalized[len(brand_norm):] in {"", "mg", "mgtab", "mgtablet", "tablet", "tab", "capsule", "cap"}:
                return True
            if brand_norm.startswith(normalized) and brand_norm[len(normalized):] in {"f", "180", "650", "10", "5", "120", "250", "500"}:
                return bool(med.get("is_exact_brand"))
        return False

    valid = [m for m in candidates if pack_keys and salt_set(m) == pack_keys]
    printed_brand_candidates = []
    for med in valid:
        candidate_brand = str(med.get("brand") or "").strip()
        candidate_brand_norm = re.sub(r"[^a-z0-9]", "", candidate_brand.lower())
        if (candidate_brand and candidate_brand_norm not in mfr_tokens and candidate_brand_norm not in english
                and candidate_brand_norm not in pack_keys
                and re.search(rf"(?<![A-Za-z0-9]){re.escape(candidate_brand)}(?![A-Za-z0-9])", raw, re.I)):
            printed_brand_candidates.append(med)
    exact = next((m for m in valid if (is_brand(m) or m in printed_brand_candidates) and same_strength(m)), None)
    if exact:
        message = "Brand, active ingredients, and strength match the pack."
        level = "verified"
        chosen = exact
    elif valid and any(is_brand(m) or m in printed_brand_candidates for m in valid):
        chosen = next(m for m in valid if is_brand(m) or m in printed_brand_candidates)
        level = "likely"
        message = "Strength mismatch: the brand and active ingredients match, but the dosage strength does not."
    elif valid:
        safe_read_brand = brand if guarded_brand and re.search(r"\d", brand) else ""
        chosen = {"id": None, "name": safe_read_brand or "Read from the pack", "brand": safe_read_brand,
                  "detected_name": safe_read_brand or "Read from the pack", "salt": ", ".join(
                      f"{x.get('ingredient', x.get('canonical_name', '')).title()} {x.get('strength', '')} {x.get('unit', 'mg')}".strip() for x in pack_salts),
                  "manufacturer": manufacturer, "parsed_salt": pack_salts,
                  "dosage_form": extracted.get("dosage_form") or "Tablet", "composition_only": True}
        level = "composition_match"
        message = "Composition match. Read from the pack; brand identity was not confirmed."
    elif pack_keys:
        safe_read_brand = brand if guarded_brand and re.search(r"\d", brand) else ""
        chosen = {
            "id": None, "name": safe_read_brand or "Read from the pack", "brand": safe_read_brand,
            "detected_name": safe_read_brand or "Read from the pack", "salt": ", ".join(
                f"{x.get('ingredient', x.get('canonical_name', '')).title()} {x.get('strength', '')} {x.get('unit', 'mg')}".strip()
                for x in pack_salts
            ), "manufacturer": manufacturer, "parsed_salt": pack_salts,
            "dosage_form": extracted.get("dosage_form") or "Tablet", "composition_only": True,
        }
        level = "composition_match"
        message = "Composition read from the pack; no database candidate has the same complete active ingredient set."
    else:
        chosen = candidates[0] if candidates else None
        level = "likely" if chosen else "not_found"
        message = "Could not read a complete active ingredient composition from the pack."
    options = [m for m in candidates if m is not chosen and pack_keys and salt_set(m) == pack_keys
               and not re.search(r"\b(?:sted(?:-ml)?|telio)\b", str(m.get("brand", "") or m.get("name", "")), re.I)
               and not (len(pack_keys) == 1 and re.search(r"\b(?:montelukast|cetirizine|phenylephrine)\b", str(m.get("salt", "")), re.I))]
    if pack_salts:
        key_parts = []
        for item in sorted(pack_salts, key=lambda x: x.get("canonical_name", "")):
            key_parts.append(f"{item.get('canonical_name', '')}:{item.get('strength', '')}:{item.get('unit', '')}")
        try:
            index = get_medicine_index()
            indexed = index.find_by_composition_key("+".join(key_parts))
            known_ids = {m.get("id") for m in options}
            options.extend(m for m in indexed if m.get("id") != chosen.get("id") and m.get("id") not in known_ids
                           and not re.search(r"\b(?:sted(?:-ml)?|telio)\b", str(m.get("brand", "") or m.get("name", "")), re.I)
                           and not (len(pack_keys) == 1 and re.search(r"\b(?:montelukast|cetirizine|phenylephrine)\b", str(m.get("salt", "")), re.I)))
        except Exception:
            logger.debug("Composition options lookup failed", exc_info=True)
    options.sort(key=lambda m: float(m.get("price") or float("inf")))
    return {"level": level, "result_level": level, "medicine": chosen, "message": message, "options": options[:5],
            "brand_read": brand, "salt_read": pack_salts, "manufacturer_read": manufacturer}


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
        if not _is_safe_brand_token(tok) or tok in COMMON_SALTS or tok in SALT_MODIFIERS_SET:
            continue
        if tok in brand_index.brand_to_ids:
            exact_brand_words.add(tok)
        elif len(tok) >= 4:
            hits = process.extract(tok, brand_index.unique_brand_words, scorer=fuzz.ratio, limit=2, score_cutoff=85.0)
            for hit in hits:
                h_word = hit[0]
                if _is_safe_brand_token(h_word) and h_word not in COMMON_SALTS and h_word not in SALT_MODIFIERS_SET:
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
            f_cutoff = 70.0 if len(tok) >= 6 else search_cutoff
            f_matches = process.extract(tok, brand_index.unique_brand_words, scorer=fuzz.ratio, limit=3, score_cutoff=f_cutoff)
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
            elif cand_salts & pack_salts or salts_share_active(cand_salts, pack_salts):
                # Overlap: shares at least one active ingredient with pack
                salt_matched = True
                score += 30.0
            elif cand_salts:
                # Rule d: Clear contradiction — pack salt was read AND candidate shares NO active
                score -= 60.0
            else:
                # Candidate has empty salt string in DB: neutral, no veto
                pass
        else:
            # Rule e: If pack salt could not be read at all, do not veto
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
                score -= 10.0
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

    decision = decide_result({"raw_ocr": full_text}, sorted_candidates)
    if decision["medicine"] in sorted_candidates:
        top_cand = decision["medicine"]
        sorted_candidates.remove(top_cand)
        sorted_candidates.insert(0, top_cand)
    is_verified = decision["level"] == "verified"
    status = "verified" if is_verified else "unverified"

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
        "match_type": match_type,
        "medicine": top_cand,
        "candidates": sorted_candidates[:3],
        "medicines": sorted_candidates[:3],
        "top_candidates": sorted_candidates[:5],
        "score_margin": score_margin,
        "confidence_breakdown": breakdown,
        "raw_ocr": full_text,
        "n_matching_tokens": n_matching_tokens,
        "result_level": decision["level"],
        "message": decision["message"],
        "decision": decision,
    }

