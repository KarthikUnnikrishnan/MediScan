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
    '5CMX5YD', 'NAN', 'TAPE', 'OINT'
}

COMMON_SALTS = {
    'PARACETAMOL', 'ACETAMINOPHEN', 'IBUPROFEN', 'TRETINOIN', 'ACETYLSALICYLIC', 'ASPIRIN',
    'AMOXICILLIN', 'AMOXYCILLIN', 'CLAVULANIC', 'CLAVULANATE', 'CETIRIZINE', 'LEVOCETIRIZINE',
    'AZITHROMYCIN', 'DICLOFENAC', 'ACECLOFENAC', 'PANTOPRAZOLE', 'OMEPRAZOLE', 'RABEPRAZOLE',
    'METFORMIN', 'GLIMEPIRIDE', 'ATORVASTATIN', 'ROSUVASTATIN', 'AMLODIPINE', 'TELMISARTAN',
    'SALBUTAMOL', 'ALBUTEROL', 'POVIDONE', 'IODINE', 'BETAMETHASONE', 'CELECOXIB', 'METRONIDAZOLE',
    'LACTULOSE', 'TERBINAFINE', 'DOMPERIDONE', 'FUSIDIC', 'GENTAMICIN', 'SITAGLIPTIN', 'ACYCLOVIR'
}


def split_glued_tokens(text: str) -> str:
    """Split glued tokens at letter/digit boundaries (e.g. ACRETIN0.05% -> ACRETIN 0.05%)."""
    t = re.sub(r'([A-Za-z]+)(\d+(?:\.\d+)?%?)', r'\1 \2', text)
    t = re.sub(r'(\d+(?:\.\d+)?%?)([A-Za-z]+)', r'\1 \2', t)
    return t


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

    full_text = " ".join(lines).upper()
    detected_strengths = re.findall(r'\b\d+(?:\.\d+)?%?\b', full_text)

    # Candidate tokens with box prominence weights (larger text preferred)
    candidate_tokens: Dict[str, Tuple[float, float, int]] = {}
    for box_idx, line in enumerate(lines):
        prominence = 20.0 if box_idx == 0 else (12.0 if box_idx == 1 else (6.0 if box_idx == 2 else 0.0))
        cleaned = split_glued_tokens(line.upper())
        words = [re.sub(r'[^A-Z0-9]', '', w) for w in re.split(r'[\s\-_]+', cleaned) if w.strip()]

        # 1. Word alone
        for w in words:
            if len(w) >= 3 and w not in PHARMA_STOPWORDS and not w.isdigit():
                cur_sc, cur_prom, cur_cnt = candidate_tokens.get(w, (0.0, 0.0, 0))
                candidate_tokens[w] = (cur_sc, max(cur_prom, prominence), cur_cnt + 1)

        # 2. Hyphen joined
        hyphen_j = re.sub(r'[^A-Z0-9]', '', line.upper())
        if len(hyphen_j) >= 3 and hyphen_j not in PHARMA_STOPWORDS and not hyphen_j.isdigit():
            cur_sc, cur_prom, cur_cnt = candidate_tokens.get(hyphen_j, (0.0, 0.0, 0))
            candidate_tokens[hyphen_j] = (cur_sc, max(cur_prom, prominence), cur_cnt + 1)

        # 3. Two-word joins
        for i in range(len(words) - 1):
            two_j = words[i] + words[i + 1]
            if len(two_j) >= 4 and two_j not in PHARMA_STOPWORDS and not two_j.isdigit():
                cur_sc, cur_prom, cur_cnt = candidate_tokens.get(two_j, (0.0, 0.0, 0))
                candidate_tokens[two_j] = (cur_sc, max(cur_prom, prominence), cur_cnt + 1)

    brand_scores: Dict[str, Tuple[float, int]] = {}
    for tok, (_, prom, count) in candidate_tokens.items():
        if tok in brand_index.brand_to_ids:
            score = 100.0 + prom
            if tok in COMMON_SALTS:
                score -= 15.0  # slight penalty if token is a generic chemical salt
            if tok not in brand_scores or score > brand_scores[tok][0]:
                brand_scores[tok] = (score, count)
        elif len(tok) >= 4:
            f_matches = process.extract(tok, brand_index.unique_brand_words, scorer=fuzz.ratio, limit=3, score_cutoff=search_cutoff)
            for m_b, score, _ in f_matches:
                adj_score = score + prom
                if m_b in COMMON_SALTS:
                    adj_score -= 15.0
                if m_b not in brand_scores or adj_score > brand_scores[m_b][0]:
                    brand_scores[m_b] = (adj_score, count)

    if not brand_scores:
        return {
            "success": False,
            "status": "unverified",
            "medicine": None,
            "candidates": [],
            "medicines": [],
            "top_candidates": [],
            "confidence_breakdown": _zero_breakdown(),
            "raw_ocr": full_text,
            "message": "No matching brand words found in local database."
        }

    # Gather candidate medicine rows from SQLite
    cand_ids = []
    brand_by_id = {}
    for b_word, (b_score, b_count) in sorted(brand_scores.items(), key=lambda x: x[1][0], reverse=True)[:10]:
        for mid in brand_index.brand_to_ids.get(b_word, [])[:15]:
            cand_ids.append(mid)
            brand_by_id[mid] = (b_word, b_score, b_count)

    rows_db = brand_index.get_medicines_by_ids(cand_ids)

    scored_candidates = []
    for row in rows_db:
        mid = row["id"]
        name = row["name"]
        salt = row.get("salt") or ""
        mfr = row.get("manufacturer") or ""
        price = row.get("price")

        b_word, b_score, b_count = brand_by_id.get(mid, ("", 0.0, 1))
        cand_name = (name or "").upper()
        cand_salt = salt.upper()
        cand_mfr = mfr.upper()

        score = b_score

        # 1. Salt boost (+35)
        salt_matched = False
        salt_words = [re.sub(r'[^A-Z0-9]', '', w) for w in re.split(r'[\s,()/+]+', cand_salt) if len(w) >= 4]
        if any(sw in full_text and sw not in PHARMA_STOPWORDS for sw in salt_words):
            salt_matched = True
            score += 35.0

        # 2. Strength corroboration (+15 / -10)
        cand_strengths = re.findall(r'\b\d+(?:\.\d+)?%?\b', cand_name + ' ' + cand_salt)
        strength_matched = False
        if any(st in cand_strengths for st in detected_strengths if len(st) >= 2 or st in {'5', '2'}):
            strength_matched = True
            score += 15.0
        elif detected_strengths and cand_strengths:
            score -= 10.0

        # 3. Manufacturer boost (+10)
        if cand_mfr:
            mfr_words = [w for w in re.split(r'[\s,.]+', cand_mfr) if len(w) >= 4 and w not in PHARMA_STOPWORDS]
            if any(mw in full_text for mw in mfr_words):
                score += 10.0

        # 4. Multi-line support (+3 per extra line)
        if b_count > 1:
            score += min(9.0, (b_count - 1) * 3.0)

        # 5. Compound-count guard (-20)
        if '+' in cand_salt:
            score -= 20.0

        corroborated = bool(salt_matched or strength_matched)
        confident = bool(score >= cutoff)

        first_brand = re.split(r'[\s\-_]+', name.strip())[0]
        scored_candidates.append({
            "id": mid,
            "name": name,
            "brand": first_brand,
            "salt": salt,
            "manufacturer": mfr or "Registered Manufacturer",
            "price": price,
            "score": round(score, 1),
            "confident": confident,
            "corroborated": corroborated,
            "salt_matched": salt_matched,
            "strength_matched": strength_matched,
            "dosage_form": "Tablet"
        })

    # Deduplicate candidates by brand name, keeping highest scoring row
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    dedup_candidates = []
    seen_brands = set()
    for cand in scored_candidates:
        b_norm = re.sub(r'[^A-Z0-9]', '', cand["brand"].upper())
        if b_norm not in seen_brands:
            seen_brands.add(b_norm)
            dedup_candidates.append(cand)
            if len(dedup_candidates) >= 5:
                break

    if not dedup_candidates:
        return {
            "success": False,
            "status": "unverified",
            "medicine": None,
            "candidates": [],
            "medicines": [],
            "top_candidates": [],
            "confidence_breakdown": _zero_breakdown(),
            "raw_ocr": full_text,
            "message": "Unable to find candidate medicines in local database."
        }

    # Requirement A: The result status is "verified" only when the top candidate
    # is confident AND corroborated (salt or strength matches); otherwise status "unverified".
    top_cand = dedup_candidates[0]
    is_verified = bool(top_cand.get("confident") and top_cand.get("corroborated"))
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
        "medicine": top_cand if is_verified else None,
        "candidates": dedup_candidates[:3],
        "medicines": dedup_candidates[:3],
        "top_candidates": dedup_candidates[:5],
        "confidence_breakdown": breakdown,
        "raw_ocr": full_text,
        "message": None if is_verified else "Top candidate was not confident and corroborated. Top candidate suggestions provided."
    }

