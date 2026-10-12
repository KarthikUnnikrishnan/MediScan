"""
scanner/ml/alternative_matcher.py - composition-aware cheaper alternative search.

Alternatives are only shown when the source medicine and candidate both have
complete, matching active composition and comparable price data.
"""

import logging
import sqlite3
from typing import Any, Dict, List, Tuple

from django.conf import settings

from .medicine_matcher import are_forms_compatible
from .salt_parser import normalize_ingredient_name, parse_salt

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
) -> List[Dict[str, Any]]:
    """
    Find cheaper equivalent medicines.

    A source medicine with incomplete salt/composition data gets no alternatives.
    This prevents unrelated results caused by matching a single word or ingredient.
    """
    if not verified_medicine:
        return []

    med_id = verified_medicine.get("id")
    med_name = (verified_medicine.get("name") or "").strip()
    raw_salt = (verified_medicine.get("salt") or "").strip()
    primary_price = verified_medicine.get("price")
    primary_form = verified_medicine.get("dosage_form")

    parsed_salt = verified_medicine.get("parsed_salt") or parse_salt(raw_salt)
    target_comp = _normalize_composition_list(parsed_salt)
    if not _composition_is_complete(target_comp):
        logger.info(
            "No alternatives for %r: source composition is incomplete.",
            med_name,
        )
        return []

    if not primary_price or primary_price <= 0:
        logger.info("No alternatives for %r: source price is not comparable.", med_name)
        return []

    keyword = target_comp[0][0]

    conn = None
    try:
        conn = sqlite3.connect(str(settings.MEDICINES_DB_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, salt, manufacturer, price "
            "FROM medicines "
            "WHERE LOWER(salt) LIKE ? "
            "  AND price IS NOT NULL AND price > 0 AND price < ? "
            "  AND id != ? "
            "ORDER BY price ASC "
            "LIMIT 1000",
            (f"%{keyword}%", primary_price, med_id),
        )

        rows = cur.fetchall()
        validated_alternatives = []
        seen_brands = {med_name.lower()}

        for row in rows:
            cid = row["id"]
            cname = (row["name"] or "").strip()
            csalt = row["salt"] or ""
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

            cand_form = _dosage_form_from_name(cname)
            form_compatible, _ = are_forms_compatible(primary_form, cand_form)
            if not form_compatible:
                continue

            if not cprice or cprice >= primary_price:
                continue

            diff = round(primary_price - cprice, 2)
            pct = round((diff / primary_price) * 100)

            seen_brands.add(cname_key)
            validated_alternatives.append({
                "id": cid,
                "name": cname,
                "manufacturer": cmfr or "Manufacturer unavailable",
                "salt": csalt,
                "price": round(cprice, 2),
                "saving_amount": diff,
                "saving_percent": pct,
                "same_composition": True,
                "has_savings_data": True,
                "validation": {
                    "same_ingredients": True,
                    "same_strengths": True,
                    "same_form": True,
                    "cheaper": True,
                    "price": round(cprice, 2),
                },
            })

            if len(validated_alternatives) >= limit:
                break

        logger.info(
            "find_top_cheaper_alternatives(%r): %d validated from %d candidates",
            med_name,
            len(validated_alternatives),
            len(rows),
        )
        return validated_alternatives

    except Exception as exc:
        logger.error("find_top_cheaper_alternatives failed: %s", exc, exc_info=True)
        return []
    finally:
        if conn:
            conn.close()
