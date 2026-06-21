"""
Model 4 — Side Effects + Drug Interaction Checker
Sources: SIDER 4.1 (FDA) + DrugBank 191K pairs + TWOSIDES 42M pairs
"""

import sqlite3, os
from rapidfuzz import process, fuzz

DB_PATH = r"D:\Coding Section\Mediscan\db_cache\drugs.sqlite"

_conn       = None
_sider_names = []   # drug names in SIDER map
_ddi_names   = []   # drug names in DrugBank


def load_db():
    global _conn, _sider_names, _ddi_names
    _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    _conn.row_factory = sqlite3.Row

    _sider_names = [r[0] for r in _conn.execute(
        "SELECT DISTINCT drug_name FROM drug_stitch_map"
    ).fetchall() if r[0]]

    _ddi_names = list(set(
        [r[0] for r in _conn.execute(
            "SELECT DISTINCT drug1 FROM drug_interactions LIMIT 50000"
        ).fetchall() if r[0]] +
        [r[0] for r in _conn.execute(
            "SELECT DISTINCT drug2 FROM drug_interactions LIMIT 50000"
        ).fetchall() if r[0]]
    ))

    print(f"[DrugInfo] SIDER names: {len(_sider_names):,}")
    print(f"[DrugInfo] DDI names  : {len(_ddi_names):,}")
    print(f"[DrugInfo] DB loaded")


def _fuzzy_match(name, pool, threshold=65):
    if not pool:
        return None, 0
    result = process.extractOne(name, pool, scorer=fuzz.WRatio)
    if result and result[1] >= threshold:
        return result[0], result[1]
    return None, 0


def get_side_effects(drug_name: str) -> dict:
    if _conn is None:
        return {"success": False, "drug_name": drug_name,
                "side_effects": [], "error": "DB not loaded."}

    matched, score = _fuzzy_match(drug_name, _sider_names, threshold=55)

    if matched:
        stitch_row = _conn.execute(
            "SELECT stitch_id FROM drug_stitch_map WHERE drug_name = ? LIMIT 1",
            (matched,)
        ).fetchone()

        if stitch_row:
            stitch_id = stitch_row['stitch_id']

            # Try frequency table first
            rows = _conn.execute("""
                SELECT se_name, freq_label
                FROM   se_frequency
                WHERE  stitch_id = ?
                  OR   stitch_id = REPLACE(?, 'CID0', 'CID')
                ORDER  BY CASE freq_label
                            WHEN 'Common'   THEN 1
                            WHEN 'Uncommon' THEN 2
                            WHEN 'Rare'     THEN 3
                            ELSE 4
                          END
                LIMIT 3
            """, (stitch_id, stitch_id)).fetchall()

            if rows:
                return {
                    "success"      : True,
                    "drug_name"    : drug_name,
                    "matched_name" : matched,
                    "side_effects" : [
                        {"name": r['se_name'], "frequency": r['freq_label']}
                        for r in rows
                    ],
                    "error": "",
                }

            # Fallback: all_se table (no frequency label)
            rows = _conn.execute("""
                SELECT DISTINCT se_name FROM side_effects
                WHERE  stitch_id = ?
                  OR   stitch_id = REPLACE(?, 'CID0', 'CID')
                LIMIT 3
            """, (stitch_id, stitch_id)).fetchall()

            if rows:
                return {
                    "success"      : True,
                    "drug_name"    : drug_name,
                    "matched_name" : matched,
                    "side_effects" : [
                        {"name": r['se_name'], "frequency": "Unknown"}
                        for r in rows
                    ],
                    "error": "",
                }

    return {
        "success"     : False,
        "drug_name"   : drug_name,
        "side_effects": [],
        "error"       : f"No side effect data found for '{drug_name}'.",
    }


def check_interactions(drug1: str, drug2: str) -> dict:
    if _conn is None:
        return {"success": False, "interacts": False, "error": "DB not loaded."}

    # Try multiple name variants for each drug
    def get_variants(name):
        variants = [name, name.lower()]
        # Common brand → generic mappings
        aliases = {
            "aspirin"    : "acetylsalicylic acid",
            "paracetamol": "acetaminophen",
            "brufen"     : "ibuprofen",
            "crocin"     : "paracetamol",
            "dolo"       : "paracetamol",
        }
        lower = name.lower()
        if lower in aliases:
            variants.append(aliases[lower])
        # Also try first word only (generic name often first)
        first_word = name.split()[0]
        if len(first_word) > 3:
            variants.append(first_word)
        return variants

    # Match drug1 and drug2 against DDI database
    m1, m2 = None, None
    for v in get_variants(drug1):
        m1, s1 = _fuzzy_match(v, _ddi_names, threshold=60)
        if m1:
            break
    for v in get_variants(drug2):
        m2, s2 = _fuzzy_match(v, _ddi_names, threshold=60)
        if m2:
            break

    if not m1 or not m2:
        missing = drug1 if not m1 else drug2
        return {
            "success": True, "drug1": drug1, "drug2": drug2,
            "interacts": False,
            "description": f"'{missing}' not found in interaction database.",
            "severity": "Unknown", "error": "",
        }

    row = _conn.execute("""
        SELECT description FROM drug_interactions
        WHERE (LOWER(drug1) = LOWER(?) AND LOWER(drug2) = LOWER(?))
           OR (LOWER(drug1) = LOWER(?) AND LOWER(drug2) = LOWER(?))
        LIMIT 1
    """, (m1, m2, m2, m1)).fetchone()

    if not row:
        return {
            "success": True, "drug1": drug1, "drug2": drug2,
            "interacts": False,
            "description": "No known interaction between these medicines.",
            "severity": "None", "error": "",
        }

    desc = row['description']
    desc_lower = desc.lower()
    if any(w in desc_lower for w in ['severe','fatal','contraindicated',
                                      'avoid','dangerous','death']):
        severity = "High"
    elif any(w in desc_lower for w in ['increase','decrease','risk',
                                        'caution','monitor','bleeding']):
        severity = "Moderate"
    else:
        severity = "Low"

    return {
        "success": True, "drug1": drug1, "drug2": drug2,
        "interacts": True, "description": desc,
        "severity": severity, "error": "",
    }