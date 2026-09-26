"""
scanner/ml/drug_safety.py — Clinical safety, adverse effects, and drug interaction engine.

Uses verified active chemical ingredients to query saved_models/drugs.sqlite:
- Translates INN / British Approved Names to USAN/STITCH dictionary
- Side effects with MedDRA classification & frequency labels
- Drug-drug interactions with HIGH / MODERATE / LOW risk severity
"""

import sqlite3
import logging
import re
from typing import List, Dict, Any, Optional, Set
from django.conf import settings

from .salt_parser import normalize_ingredient_name

logger = logging.getLogger(__name__)

# International synonym mapping to match drug_stitch_map names in drugs.sqlite
_STITCH_SYNONYMS = {
    'paracetamol': 'acetaminophen',
    'amoxycillin': 'amoxicillin',
    'amoxicillin': 'amoxicillin',
    'levocetirizine': 'cetirizine',
    'cetirizine': 'cetirizine',
    'salbutamol': 'albuterol',
    'frusemide': 'furosemide',
    'furosemide': 'furosemide',
    'diclofenac': 'diclofenac',
    'azithromycin': 'azithromycin',
    'clavulanic acid': 'clavulanic',
    'clavulanate': 'clavulanic',
    'pantoprazole': 'pantoprazole',
    'omeprazole': 'omeprazole',
    'rabeprazole': 'rabeprazole',
    'metformin': 'metformin',
    'glimepiride': 'glimepiride',
    'atorvastatin': 'atorvastatin',
    'rosuvastatin': 'rosuvastatin',
    'aspirin': 'aspirin',
    'ibuprofen': 'ibuprofen',
    'aceclofenac': 'aceclofenac',
    'chlorzoxazone': 'chlorzoxazone',
    'caffeine': 'caffeine',
    'montelukast': 'montelukast',
    'fexofenadine': 'fexofenadine',
    'telmisartan': 'telmisartan',
    'amlodipine': 'amlodipine',
}

# Interaction severity classifiers
_HIGH_SEVERITY_WORDS = re.compile(
    r'\b(?:severe|fatal|avoid|contraindicated|toxicity|bleeding|arrhythmia|cardiac arrest|hemorrhage|respiratory depression|coma)\b',
    re.IGNORECASE
)
_MODERATE_SEVERITY_WORDS = re.compile(
    r'\b(?:increase|decrease|reduce|enhance|elevate|monitor|caution|adjust|adverse|prolong|risk)\b',
    re.IGNORECASE
)


def map_to_stitch_name(ingredient_canonical: str) -> str:
    """Map canonical ingredient to STITCH database name."""
    norm = normalize_ingredient_name(ingredient_canonical)
    return _STITCH_SYNONYMS.get(norm, norm)


def classify_severity(description: str) -> str:
    """Determine clinical interaction risk severity."""
    if _HIGH_SEVERITY_WORDS.search(description):
        return "HIGH"
    if _MODERATE_SEVERITY_WORDS.search(description):
        return "MODERATE"
    return "LOW"


def get_drug_safety_info(ingredients: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fetch comprehensive clinical safety information for a list of active ingredients.

    Parameters
    ----------
    ingredients : list of dict
        [{"ingredient": "...", "canonical_name": "...", "strength": ...}, ...]

    Returns
    -------
    dict:
        {
            "side_effects": [
                {"name": "Nausea", "frequency": "common", "ingredient": "paracetamol"}, ...
            ],
            "interactions": [
                {"drug1": "...", "drug2": "...", "description": "...", "severity": "HIGH|MODERATE|LOW"}, ...
            ]
        }
    """
    if not ingredients:
        return {"side_effects": [], "interactions": []}

    target_compounds = []
    for ing in ingredients:
        cname = ing.get("canonical_name") or normalize_ingredient_name(ing.get("ingredient", ""))
        stitch_name = map_to_stitch_name(cname)
        if stitch_name:
            target_compounds.append((cname, stitch_name))

    if not target_compounds:
        return {"side_effects": [], "interactions": []}

    conn = None
    side_effects_out = []
    interactions_out = []
    seen_se_names: Set[str] = set()
    seen_interactions: Set[str] = set()

    try:
        conn = sqlite3.connect(str(settings.DRUGS_DB_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        for original_name, stitch_name in target_compounds:
            # 1. Lookup stitch_id
            cur.execute(
                "SELECT stitch_id, drug_name FROM drug_stitch_map "
                "WHERE LOWER(drug_name) = ? OR LOWER(drug_name) LIKE ? "
                "LIMIT 2",
                (stitch_name.lower(), f"%{stitch_name.lower()}%")
            )
            stitch_rows = cur.fetchall()

            for srow in stitch_rows:
                stitch_id = srow["stitch_id"]

                # 2. Get side effects with frequency labels
                cur.execute(
                    "SELECT se.se_name, sf.freq_label "
                    "FROM side_effects se "
                    "LEFT JOIN se_frequency sf "
                    "  ON se.stitch_id = sf.stitch_id "
                    "  AND se.se_name = sf.se_name "
                    "WHERE se.stitch_id = ? "
                    "LIMIT 25",
                    (stitch_id,)
                )
                for r in cur.fetchall():
                    se_name = (r["se_name"] or "").strip()
                    freq = (r["freq_label"] or "").strip()
                    if se_name and se_name.lower() not in seen_se_names:
                        seen_se_names.add(se_name.lower())
                        side_effects_out.append({
                            "name": se_name,
                            "frequency": freq or "Recorded clinical effect",
                            "ingredient": original_name.capitalize()
                        })

            # 3. Look up interactions for this drug
            cur.execute(
                "SELECT drug1, drug2, description FROM drug_interactions "
                "WHERE LOWER(drug1) LIKE ? OR LOWER(drug2) LIKE ? "
                "LIMIT 10",
                (f"%{stitch_name.lower()}%", f"%{stitch_name.lower()}%")
            )
            for r in cur.fetchall():
                d1 = r["drug1"].strip()
                d2 = r["drug2"].strip()
                desc = (r["description"] or "").strip()
                pair_key = f"{min(d1.lower(), d2.lower())}:{max(d1.lower(), d2.lower())}"

                if pair_key not in seen_interactions:
                    seen_interactions.add(pair_key)
                    severity = classify_severity(desc)
                    interactions_out.append({
                        "drug1": d1,
                        "drug2": d2,
                        "description": desc,
                        "severity": severity
                    })

        # Sort interactions: HIGH first, then MODERATE, then LOW
        severity_order = {"HIGH": 0, "MODERATE": 1, "LOW": 2}
        interactions_out.sort(key=lambda x: severity_order.get(x["severity"], 3))

        return {
            "side_effects": side_effects_out[:20],
            "interactions": interactions_out[:12]
        }

    except Exception as exc:
        logger.error("get_drug_safety_info failed: %s", exc, exc_info=True)
        return {"side_effects": [], "interactions": []}

    finally:
        if conn:
            conn.close()
