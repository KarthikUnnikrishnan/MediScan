"""
scanner/ml/medicine_index.py — Dynamic, high-performance medicine search index.

Generates normalized searchable aliases for all 503k+ medicines in medicines.sqlite:
- Brand family, name without spaces, name without dosage forms, name + strength.
- Caches index persistently to saved_models/medicine_index.pkl for sub-second startup.
"""

import os
import re
import pickle
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple

from django.conf import settings
from .salt_parser import parse_salt, is_corrupted_salt, normalize_ingredient_name

logger = logging.getLogger(__name__)

CACHE_FILE_NAME = "medicine_index.pkl"

DOSAGE_WORDS = {
    'tablet', 'tablets', 'capsule', 'capsules', 'syrup', 'suspension',
    'injection', 'ointment', 'gel', 'cream', 'drops', 'solution',
    'infusion', 'lotion', 'inhaler', 'respules', 'spray', 'dt', 'sr',
    'er', 'cr', 'xl', 'od', 'forte', 'plus', 'liquid'
}


def normalize_string(text: Optional[str]) -> str:
    """Normalize text: lowercase, remove punctuation, single spaces."""
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


class MedicineIndex:
    """
    In-memory searchable index for medicines.sqlite.
    """

    def __init__(self):
        # med_id -> dict(id, name, salt, manufacturer, price, parsed_salt, dosage_form)
        self.medicines_by_id: Dict[int, Dict[str, Any]] = {}

        # alias -> list of med_ids (ranked best first)
        self.alias_to_ids: Dict[str, List[int]] = {}

        # Set of unique brand families for fuzzy matching
        self.unique_brand_names: List[str] = []

        # Canonical ingredient combination -> list of med_ids (for bioequivalence)
        # e.g. "amoxycillin:500:mg+clavulanic acid:125:mg" -> [id1, id2, ...]
        self.composition_to_ids: Dict[str, List[int]] = {}

        self.is_loaded = False

    def get_medicine(self, med_id: int) -> Optional[Dict[str, Any]]:
        return self.medicines_by_id.get(med_id)

    def search_alias(self, query: str) -> List[Dict[str, Any]]:
        """
        Look up exact alias candidates.
        """
        if not query:
            return []
        
        norm = normalize_string(query)
        nospace = norm.replace(" ", "")

        matched_ids: List[int] = []
        # Try spaced alias first, then nospace
        if norm in self.alias_to_ids:
            matched_ids = self.alias_to_ids[norm]
        elif nospace in self.alias_to_ids:
            matched_ids = self.alias_to_ids[nospace]

        results = []
        for mid in matched_ids:
            med = self.medicines_by_id.get(mid)
            if med:
                results.append(med)
        return results

    def find_by_composition_key(self, comp_key: str) -> List[Dict[str, Any]]:
        mids = self.composition_to_ids.get(comp_key, [])
        return [self.medicines_by_id[mid] for mid in mids if mid in self.medicines_by_id]


_GLOBAL_INDEX: Optional[MedicineIndex] = None


def _extract_dosage_form_from_name(name: str) -> Optional[str]:
    name_l = name.lower()
    for w in ['tablet', 'capsule', 'syrup', 'suspension', 'injection', 'ointment', 'gel', 'cream', 'drops']:
        if w in name_l:
            return w.capitalize()
    return None


def _build_composition_key(parsed_salt: List[Dict[str, Any]]) -> str:
    """Build a deterministic sorted key for active composition and strength."""
    if not parsed_salt:
        return ""
    sorted_items = sorted(
        parsed_salt,
        key=lambda x: x.get('canonical_name', '')
    )
    parts = []
    for item in sorted_items:
        cname = item.get('canonical_name', '')
        st = item.get('strength', '')
        unit = item.get('unit', '')
        parts.append(f"{cname}:{st}:{unit}")
    return "+".join(parts)


def build_and_cache_index(db_path: Path, cache_path: Path) -> MedicineIndex:
    """
    Build index from database and save to pickle file.
    """
    logger.info("Building MedicineIndex from %s...", db_path)
    index = MedicineIndex()

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT id, name, salt, manufacturer, price FROM medicines")
    rows = cur.fetchall()
    conn.close()

    alias_dict: Dict[str, List[Tuple[int, int]]] = {}
    brand_set: Set[str] = set()

    for row in rows:
        med_id, name, salt, manufacturer, price = row
        if not name or not name.strip():
            continue

        clean_name = name.strip()
        clean_mfr = (manufacturer or "").strip()
        has_valid_salt = not is_corrupted_salt(salt)
        has_mfr = bool(clean_mfr)
        has_price = bool(price and price > 0)

        # Quality rank (higher is better):
        # Valid salt + valid mfr + valid price = score 7
        quality_score = (4 if has_valid_salt else 0) + (2 if has_mfr else 0) + (1 if has_price else 0)

        parsed_salts = parse_salt(salt) if has_valid_salt else []
        dosage_form = _extract_dosage_form_from_name(clean_name)
        comp_key = _build_composition_key(parsed_salts)

        record = {
            "id": med_id,
            "name": clean_name,
            "salt": salt or "",
            "parsed_salt": parsed_salts,
            "manufacturer": clean_mfr,
            "price": price,
            "dosage_form": dosage_form,
            "quality_score": quality_score,
            "composition_key": comp_key,
        }
        index.medicines_by_id[med_id] = record

        # Register composition key
        if comp_key:
            if comp_key not in index.composition_to_ids:
                index.composition_to_ids[comp_key] = []
            index.composition_to_ids[comp_key].append(med_id)

        # Generate searchable aliases
        norm_name = normalize_string(clean_name)
        nospace_name = norm_name.replace(" ", "")

        words = norm_name.split()
        non_form_words = [w for w in words if w not in DOSAGE_WORDS]
        brand_family = non_form_words[0] if non_form_words else (words[0] if words else "")
        no_form_name = " ".join(non_form_words)
        no_form_nospace = no_form_name.replace(" ", "")

        if brand_family:
            brand_set.add(brand_family)

        aliases = {norm_name, nospace_name, no_form_name, no_form_nospace, brand_family}

        # Add name + strength alias if strengths found in name
        strength_match = re.search(r'\b\d+\b', norm_name)
        if strength_match and brand_family:
            aliases.add(f"{brand_family} {strength_match.group(0)}")
            aliases.add(f"{brand_family}{strength_match.group(0)}")

        for alias in aliases:
            if len(alias) >= 2:
                if alias not in alias_dict:
                    alias_dict[alias] = []
                alias_dict[alias].append((med_id, quality_score))

    # Sort aliases by quality score descending and keep top 10 per alias
    for alias, id_tuples in alias_dict.items():
        sorted_ids = [t[0] for t in sorted(id_tuples, key=lambda x: x[1], reverse=True)]
        # deduplicate while preserving order
        seen_ids = set()
        dedup_ids = []
        for mid in sorted_ids:
            if mid not in seen_ids:
                seen_ids.add(mid)
                dedup_ids.append(mid)
            if len(dedup_ids) >= 10:
                break
        index.alias_to_ids[alias] = dedup_ids

    index.unique_brand_names = sorted(brand_set)
    index.is_loaded = True

    logger.info(
        "Indexed %d medicines into %d aliases and %d brands. Saving to cache %s...",
        len(index.medicines_by_id), len(index.alias_to_ids), len(index.unique_brand_names), cache_path
    )

    try:
        with open(cache_path, "wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Successfully persisted medicine index cache to %s", cache_path)
    except Exception as exc:
        logger.warning("Failed to save medicine index cache: %s", exc)

    return index


def get_medicine_index() -> MedicineIndex:
    """
    Get or load the global singleton medicine index.
    Loads from cache file if available, else builds and caches.
    """
    global _GLOBAL_INDEX
    if _GLOBAL_INDEX is not None and _GLOBAL_INDEX.is_loaded:
        return _GLOBAL_INDEX

    db_path = Path(settings.MEDICINES_DB_PATH)
    cache_path = db_path.parent / CACHE_FILE_NAME

    if cache_path.exists():
        try:
            logger.info("Loading cached MedicineIndex from %s...", cache_path)
            with open(cache_path, "rb") as f:
                index = pickle.load(f)
            if isinstance(index, MedicineIndex) and index.medicines_by_id:
                _GLOBAL_INDEX = index
                logger.info("Loaded %d medicines from cache.", len(index.medicines_by_id))
                return _GLOBAL_INDEX
        except Exception as exc:
            logger.warning("Failed to load cached index (%s). Rebuilding...", exc)

    _GLOBAL_INDEX = build_and_cache_index(db_path, cache_path)
    return _GLOBAL_INDEX
