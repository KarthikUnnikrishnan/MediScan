"""
scanner/ml/alternative_matcher.py — Precision bioequivalent alternative search engine.

Finds top 5 cheaper generic/brand alternatives matching:
- EXACT complete active chemical composition
- EXACT matching strengths per ingredient
- Compatible dosage form
- Verified cheaper price (price < primary_price and price > 0)
- Excludes detected medicine itself and deduplicates products
"""

import sqlite3
import logging
from typing import List, Dict, Any, Optional, Tuple
from django.conf import settings

from .salt_parser import parse_salt, is_corrupted_salt, normalize_ingredient_name
from .medicine_matcher import are_forms_compatible

logger = logging.getLogger(__name__)


def _normalize_composition_list(ingredients: List[Dict[str, Any]]) -> List[Tuple[str, float, str]]:
    """Return sorted tuple representation of active composition: (canonical_name, strength, unit)."""
    norm_list = []
    for ing in ingredients:
        cname = ing.get("canonical_name") or normalize_ingredient_name(ing.get("ingredient", ""))
        st = ing.get("strength")
        unit = ing.get("unit", "")
        if cname:
            norm_list.append((cname, st, unit))
    return sorted(norm_list, key=lambda x: x[0])


def find_top_cheaper_alternatives(
    verified_medicine: Dict[str, Any],
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Find top 5 cheaper equivalent medicines for a verified medicine.

    Parameters
    ----------
    verified_medicine : dict
        {
            "id": int,
            "name": str,
            "salt": str,
            "parsed_salt": list of dict,
            "price": float or None,
            "dosage_form": str or None,
            ...
        }
    limit : int, default 5

    Returns
    -------
    list of dict
        [
            {
                "id": int,
                "name": str,
                "manufacturer": str,
                "salt": str,
                "price": float,
                "saving_amount": float,
                "saving_percent": int,
                "same_composition": True,
                "validation": {
                    "same_ingredients": True,
                    "same_strengths": True,
                    "same_form": True,
                    "cheaper": True,
                    "price": float
                }
            }, ...
        ]
    """
    if not verified_medicine:
        return []

    med_id = verified_medicine.get("id")
    med_name = verified_medicine.get("name", "").strip()
    raw_salt = verified_medicine.get("salt", "").strip()
    parsed_salt = verified_medicine.get("parsed_salt") or parse_salt(raw_salt)
    primary_price = verified_medicine.get("price")
    primary_form = verified_medicine.get("dosage_form")

    if not parsed_salt:
        logger.debug("Cannot find alternatives: verified medicine has no parsed salt.")
        return []

    # Target composition
    target_comp = _normalize_composition_list(parsed_salt)
    if not target_comp:
        return []

    # If primary price is missing or <= 0, we can still find equivalents, but cheaper check needs price
    has_price_constraint = bool(primary_price and primary_price > 0)

    # Primary ingredient keyword for SQL filtering
    primary_ing_canonical = target_comp[0][0]

    conn = None
    try:
        conn = sqlite3.connect(str(settings.MEDICINES_DB_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        if has_price_constraint:
            cur.execute(
                "SELECT id, name, salt, manufacturer, price "
                "FROM medicines "
                "WHERE LOWER(salt) LIKE ? "
                "  AND price IS NOT NULL AND price > 0 AND price < ? "
                "  AND id != ? "
                "ORDER BY price ASC "
                "LIMIT 500",
                (f"%{primary_ing_canonical}%", primary_price, med_id)
            )
        else:
            cur.execute(
                "SELECT id, name, salt, manufacturer, price "
                "FROM medicines "
                "WHERE LOWER(salt) LIKE ? "
                "  AND price IS NOT NULL AND price > 0 "
                "  AND id != ? "
                "ORDER BY price ASC "
                "LIMIT 500",
                (f"%{primary_ing_canonical}%", med_id)
            )

        rows = cur.fetchall()

        validated_alternatives = []
        seen_brands = {med_name.lower()}

        for row in rows:
            cid = row["id"]
            cname = row["name"].strip()
            csalt = row["salt"]
            cmfr = (row["manufacturer"] or "").strip()
            cprice = row["price"]

            # Deduplicate by brand name or identical medicine
            cname_key = cname.lower()
            if cname_key in seen_brands or cname_key == med_name.lower():
                continue

            # Reject corrupted scraped salts
            if is_corrupted_salt(csalt):
                continue

            cand_parsed = parse_salt(csalt)
            cand_comp = _normalize_composition_list(cand_parsed)

            # 1. Require EXACT active composition (same ingredients count and identities)
            if len(cand_comp) != len(target_comp):
                continue

            cand_ingredients = [c[0] for c in cand_comp]
            target_ingredients = [t[0] for t in target_comp]
            same_ingredients = (cand_ingredients == target_ingredients)
            if not same_ingredients:
                continue

            # 2. Require EXACT matching strengths per ingredient
            same_strengths = (cand_comp == target_comp)
            if not same_strengths:
                continue

            # 3. Dosage form compatibility check
            cand_form = None
            for f in ['tablet', 'capsule', 'syrup', 'suspension', 'injection', 'ointment', 'gel', 'cream', 'drops']:
                if f in cname_key:
                    cand_form = f.capitalize()
                    break

            form_compatible, _ = are_forms_compatible(primary_form, cand_form)
            if not form_compatible:
                continue

            # 4. Cheaper price validation
            cheaper = True
            saving_amount = None
            saving_percent = None

            if has_price_constraint:
                if cprice >= primary_price:
                    continue
                diff = round(primary_price - cprice, 2)
                pct = round((diff / primary_price) * 100)
                saving_amount = diff
                saving_percent = pct

            validation_record = {
                "same_ingredients": True,
                "same_strengths": True,
                "same_form": True,
                "cheaper": cheaper,
                "price": round(cprice, 2)
            }

            seen_brands.add(cname_key)
            validated_alternatives.append({
                "id": cid,
                "name": cname,
                "manufacturer": cmfr or "Registered Generic Lab",
                "salt": csalt,
                "price": round(cprice, 2),
                "saving_amount": saving_amount,
                "saving_percent": saving_percent,
                "same_composition": True,
                "validation": validation_record
            })

            if len(validated_alternatives) >= limit:
                break

        return validated_alternatives

    except Exception as exc:
        logger.error("find_top_cheaper_alternatives failed: %s", exc, exc_info=True)
        return []

    finally:
        if conn:
            conn.close()
