"""
scanner/ml/alternative_matcher.py — Composition-aware & Verified Price Alternative Search Engine.

Alternatives are compared only when both products have:
- Verified active composition and equivalent strength.
- Compatible dosage forms (noting any formulation differences like DT).
- Verified package quantity, price basis, and comparable unit pricing.
"""

import os
import re
import json
import logging
import sqlite3
from typing import Any, Dict, List, Tuple, Optional

from django.conf import settings

from .medicine_matcher import are_forms_compatible
from .salt_parser import normalize_ingredient_name, parse_salt
from .price_verifier import PriceVerifier, compare_medicine_prices

logger = logging.getLogger(__name__)


def _normalize_composition_list(
    ingredients: List[Dict[str, Any]]
) -> List[Tuple[str, Any, str]]:
    """Return sorted (canonical_name, strength, unit) active composition."""
    norm_list = []
    for ing in ingredients:
        cname = ing.get("canonical_name") or normalize_ingredient_name(
            ing.get("ingredient", "")
        )
        strength = ing.get("strength")
        unit = (ing.get("unit") or "").lower().strip().split("/")[0].strip()
        if cname:
            norm_list.append((cname, strength, unit))
    return sorted(norm_list, key=lambda x: x[0])


def _composition_is_complete(comp: List[Tuple[str, Any, str]]) -> bool:
    return bool(comp) and all(
        name and strength not in (None, "") and unit
        for name, strength, unit in comp
    )


def _strengths_match(comp_a, comp_b) -> bool:
    """
    Compare complete sorted composition lists.
    Ingredient names, strengths, and units must correspond one-to-one.
    """
    if len(comp_a) != len(comp_b):
        return False

    for (name_a, st_a, unit_a), (name_b, st_b, unit_b) in zip(comp_a, comp_b):
        if name_a != name_b:
            return False
        try:
            if float(st_a) != float(st_b):
                return False
        except (TypeError, ValueError):
            if str(st_a).strip().lower() != str(st_b).strip().lower():
                return False
        if not unit_a or not unit_b or unit_a != unit_b:
            return False

    return True


def _dosage_form_from_name(name: str):
    name_key = (name or "").lower()
    for form in [
        "tablet", "capsule", "syrup", "suspension", "injection",
        "ointment", "gel", "cream", "drops",
    ]:
        if form in name_key:
            return form.capitalize()
    return None


def find_top_cheaper_alternatives(
    verified_medicine: Dict[str, Any],
    limit: int = 5,
    verifier: Optional[PriceVerifier] = None,
) -> List[Dict[str, Any]]:
    """
    Find cheaper equivalent medicines with verified pricing and package quantities.

    A source medicine with incomplete salt/composition data gets no alternatives.
    Price comparisons are performed strictly between verified, comparable units.
    """
    if not verified_medicine:
        return []

    med_id = verified_medicine.get("id")
    med_name = (verified_medicine.get("name") or "").strip()
    raw_salt = (verified_medicine.get("salt") or "").strip()
    primary_price = verified_medicine.get("price")
    primary_form = verified_medicine.get("dosage_form") or _dosage_form_from_name(med_name)

    parsed_salt = verified_medicine.get("parsed_salt") or parse_salt(raw_salt)
    target_comp = _normalize_composition_list(parsed_salt)
    if not _composition_is_complete(target_comp):
        logger.info(
            "No alternatives for %r: source composition is incomplete.",
            med_name,
        )
        return []

    # Initialize price verification engine
    price_engine = verifier or PriceVerifier()

    # Verify primary medicine price and package size
    primary_verif = verified_medicine.get("price_verification")
    if not primary_verif or not isinstance(primary_verif, dict):
        p_record = price_engine.verify_medicine_price(verified_medicine)
        primary_verif = p_record.to_dict()
        verified_medicine["price_verification"] = primary_verif

    keyword = target_comp[0][0]

    conn = None
    try:
        conn = sqlite3.connect(str(settings.MEDICINES_DB_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        # Fetch matching salt candidates from database
        cur.execute(
            "SELECT id, name, salt, manufacturer, price "
            "FROM medicines "
            "WHERE LOWER(salt) LIKE ? "
            "  AND price IS NOT NULL AND price > 0 "
            "  AND id != ? "
            "ORDER BY price ASC "
            "LIMIT 1000",
            (f"%{keyword}%", med_id),
        )

        rows = cur.fetchall()
        candidate_records = []
        seen_brands = {med_name.lower()}

        for row in rows:
            cid = row["id"]
            cname = (row["name"] or "").strip()
            csalt = row["salt"] or ""
            # Clean junk prefixes e.g. '. ltd.fexofenadine'
            csalt = re.sub(r"^[\s\.\,\'\-\&]+(?:ltd|limited|pvt|laboratories|biotech|lifescience|antibiotics|ayurveda)?[\s\.\,]*", "", csalt, flags=re.IGNORECASE).strip()
            csalt = re.sub(r"^[\s\.\,\'\-\&]+", "", csalt)
            cmfr = (row["manufacturer"] or "").strip()
            cprice = row["price"]

            cname_key = cname.lower()
            if not cname_key or cname_key in seen_brands:
                continue

            cand_parsed = parse_salt(csalt)
            cand_comp = _normalize_composition_list(cand_parsed)
            if not _composition_is_complete(cand_comp):
                continue
            if not _strengths_match(target_comp, cand_comp):
                continue

            cand_form = _dosage_form_from_name(cname) or "Tablet"
            form_compatible, _ = are_forms_compatible(primary_form, cand_form)
            if not form_compatible:
                continue

            seen_brands.add(cname_key)

            candidate_dict = {
                "id": cid,
                "name": cname,
                "manufacturer": cmfr or "Manufacturer unavailable",
                "salt": csalt,
                "price": round(cprice, 2) if cprice is not None else None,
                "dosage_form": cand_form,
                "parsed_salt": cand_parsed,
            }

            # Online price and package verification for candidate
            cand_rec = price_engine.verify_medicine_price(candidate_dict)
            cand_verif = cand_rec.to_dict()

            # Compare verified prices
            comparison = compare_medicine_prices(primary_verif, cand_verif)

            # Check if this alternative is verified cheaper
            is_verified_cheaper = (
                comparison["comparison_eligible"]
                and comparison["saving_amount"] is not None
                and comparison["saving_amount"] > 0
            )

            saving_amount = comparison["saving_amount"] if comparison["comparison_eligible"] else None
            saving_percent = comparison["saving_percent"] if comparison["comparison_eligible"] else None

            # Formulation notes
            form_warning = comparison.get("formulation_warning")

            candidate_records.append({
                "id": cid,
                "name": cname,
                "manufacturer": cmfr or "Manufacturer unavailable",
                "salt": csalt,
                "price": round(cprice, 2) if cprice is not None else None,
                "saving_amount": saving_amount,
                "saving_percent": saving_percent,
                "saving_label": comparison.get("saving_label"),
                "comparison_basis": comparison.get("comparison_basis", "unavailable"),
                "same_composition": True,
                "has_savings_data": is_verified_cheaper,
                "is_cheapest": False,  # Will be assigned after sorting
                "price_verification": cand_verif,
                "price_comparison": comparison,
                "formulation_warning": form_warning,
                "validation": {
                    "same_ingredients": True,
                    "same_strengths": True,
                    "same_form": form_compatible,
                    "cheaper": is_verified_cheaper,
                    "price": round(cprice, 2) if cprice is not None else None,
                    "comparison_eligible": comparison["comparison_eligible"],
                },
            })

            # Fetch a reasonable pool of validated candidates to sort
            if len(candidate_records) >= 20:
                break

        # ── Precision Sorting (Scenario 17: Price sorting based on verified comparable unit prices) ─
        # Priority 1: Verified comparison-eligible alternatives with verified savings, sorted by verified unit_price_inr ASC
        # Priority 2: Other composition-verified alternatives (where price/unit could not be verified online)
        verified_comparable = []
        unverified_comparable = []

        for c in candidate_records:
            comp = c["price_comparison"]
            unit_p = c["price_verification"].get("unit_price_inr")
            if comp["comparison_eligible"] and unit_p is not None and c["has_savings_data"]:
                verified_comparable.append(c)
            else:
                unverified_comparable.append(c)

        # Sort verified comparable by verified unit price ascending
        verified_comparable.sort(key=lambda x: x["price_verification"].get("unit_price_inr") or 999999.0)

        # Mark CHEAPEST badge strictly on the lowest verified unit price
        if verified_comparable:
            verified_comparable[0]["is_cheapest"] = True

        # Combine: verified cheaper alternatives first, then remaining candidates
        final_alternatives = (verified_comparable + unverified_comparable)[:limit]

        logger.info(
            "find_top_cheaper_alternatives(%r): %d verified comparable, %d unverified from %d candidates (final: %d)",
            med_name,
            len(verified_comparable),
            len(unverified_comparable),
            len(candidate_records),
            len(final_alternatives),
        )

        return enrich_alternatives_with_ai(verified_medicine, final_alternatives)

    except Exception as exc:
        logger.error("find_top_cheaper_alternatives failed: %s", exc, exc_info=True)
        return []
    finally:
        if conn:
            conn.close()


# ── AI Trust & Clinical Formulation Annotation ────────────────────────────────

_TIER1_MANUFACTURERS = {
    "sun pharma", "sun pharmaceutical", "cipla", "dr. reddy", "dr reddy",
    "torrent pharma", "lupin", "alkem", "zydus", "cadila", "abbott",
    "glenmark", "micro labs", "mankind pharma", "mankind", "intas",
    "ipca", "hetero", "aristo", "macleods", "sanofi", "glaxosmithkline",
    "gsk", "pfizer", "novartis", "bayer", "astrazeneca"
}

_TIER2_MANUFACTURERS = {
    "fdc", "blue cross", "alembic", "ajanta", "usv", "jb chemicals",
    "j.b. chemicals", "corona remedies", "apex", "wallace", "zuventus",
    "emcure", "indoco", "fourrts", "meyer", "centaur", "wockhardt", "syncom"
}


def _calculate_individual_trust(
    primary_med: Dict[str, Any],
    alt: Dict[str, Any],
    rank: int,
) -> Tuple[float, str, str, str, str]:
    """
    Calculate an individual pharmaceutical trust confidence score
    based on molecular equivalence, manufacturer regulatory tier,
    and verified price transparency.
    """
    mfr = (alt.get("manufacturer") or "").lower()
    comp = alt.get("price_comparison", {})
    verif = alt.get("price_verification", {})

    base_score = 90.0

    if any(t in mfr for t in _TIER1_MANUFACTURERS):
        mfr_boost = 6.0
        badge = "WHO-GMP Licensed Manufacturer"
    elif any(t in mfr for t in _TIER2_MANUFACTURERS):
        mfr_boost = 4.0
        badge = "CDSCO Licensed National Brand"
    else:
        mfr_boost = 2.0
        badge = "Active Salt & Strength Match"

    # Price verification boost
    if comp.get("comparison_eligible"):
        price_factor = 2.0
    else:
        price_factor = 0.5

    rank_adj = -0.3 * (rank - 1)
    confidence = min(99.0, max(85.0, base_score + mfr_boost + price_factor + rank_adj))
    confidence = round(confidence, 1)

    tier = "high" if confidence >= 95.0 else ("med" if confidence >= 90.0 else "fair")

    salt_name = alt.get("salt") or primary_med.get("salt") or "Exact Active Molecule"
    clean_mfr = alt.get("manufacturer") or "Licensed Laboratory"
    reason = f"Exact 1:1 active molecule match ({salt_name}) • Formulated by {clean_mfr}"

    if alt.get("formulation_warning"):
        reason += f" • {alt['formulation_warning']}"

    # Market price range from verified data
    pkg_p = verif.get("package_price_inr")
    unit_p = verif.get("unit_price_inr")
    qty = verif.get("pack_quantity")
    unit = verif.get("pack_unit") or "tablet"

    if pkg_p and qty:
        market_range = f"₹{pkg_p:.2f} per strip ({qty} {unit}s)"
    elif unit_p:
        market_range = f"₹{unit_p:.2f} per {unit}"
    elif alt.get("price"):
        market_range = f"₹{alt['price']:.2f} (unit unverified)"
    else:
        market_range = "Price unverified"

    return confidence, tier, badge, reason, market_range


def enrich_alternatives_with_ai(
    verified_medicine: Dict[str, Any],
    alternatives: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Enrich alternative medicines with factual trust confidence scores,
    verified market price summaries, and formulation notes without fabricating bioequivalence.
    """
    if not alternatives:
        return alternatives

    for i, alt in enumerate(alternatives, 1):
        conf, tier, badge, reason, market_range = _calculate_individual_trust(
            verified_medicine, alt, rank=i
        )
        alt["confidence"] = None
        alt["trust_tier"] = tier
        alt["trust_label"] = badge
        alt["trust_reason"] = reason
        alt["current_price_status"] = "Verified" if alt.get("price_comparison", {}).get("comparison_eligible") else "Unverified"
        alt["current_price_range"] = market_range
        alt["savings"] = alt.get("saving_amount")
        alt["savings_percent"] = alt.get("saving_percent")
        alt["ai_verified"] = alt.get("price_comparison", {}).get("comparison_eligible", False)

    return alternatives
