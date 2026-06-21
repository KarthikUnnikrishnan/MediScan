"""
Model 3 — Generic Medicine Finder
Given a medicine name → find same-salt cheaper alternatives
"""

import sqlite3, os
from rapidfuzz import process, fuzz

DB_PATH = r"D:\Coding Section\Mediscan\db_cache\medicines.sqlite"

_conn        = None
_all_names   = []   # cached for fuzzy matching


def load_db():
    global _conn, _all_names
    _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    rows = _conn.execute(
        "SELECT DISTINCT name FROM medicines WHERE name IS NOT NULL"
    ).fetchall()
    _all_names = [r[0] for r in rows if r[0]]
    print(f"[GenericFinder] DB loaded — {len(_all_names):,} medicine names")


def find_generics(medicine_name: str, max_results: int = 5) -> dict:
    if _conn is None:
        return {"success": False, "input_name": medicine_name,
                "alternatives": [], "error": "DB not loaded."}

    if not medicine_name or len(medicine_name.strip()) < 2:
        return {"success": False, "input_name": medicine_name,
                "alternatives": [], "error": "Medicine name too short."}

    # Step 1: Fuzzy match to nearest DB name
    match_result = process.extractOne(
        medicine_name, _all_names, scorer=fuzz.WRatio
    )
    if not match_result or match_result[1] < 60:
        return {"success": False, "input_name": medicine_name,
                "alternatives": [],
                "error": f"'{medicine_name}' not found in database."}

    matched_name, score, _ = match_result

    # Step 2: Get salt of matched medicine
    row = _conn.execute(
        "SELECT salt FROM medicines WHERE name = ? LIMIT 1",
        (matched_name,)
    ).fetchone()

    if not row or not row['salt'] or row['salt'] in ('nan', '', 'none'):
        return {"success": False, "input_name": medicine_name,
                "matched_name": matched_name, "alternatives": [],
                "error": "Salt composition not found."}

    salt = row['salt'].strip().lower()

    # Step 3: Extract active ingredient name from salt
    # e.g. "paracetamol (650mg)" → "paracetamol"
    # e.g. "(650mg)" → search by full salt string
    import re
    drug_keyword = re.split(r'[\(\+\,]', salt)[0].strip()
    if len(drug_keyword) < 3:
        drug_keyword = salt  # fallback to full salt

    # Step 4: LIKE search — finds all doses of same drug
    rows = _conn.execute("""
        SELECT name, price, manufacturer, source, salt
        FROM   medicines
        WHERE  LOWER(salt) LIKE ?
          AND  LOWER(name) != LOWER(?)
          AND  price IS NOT NULL
          AND  price > 0
        ORDER  BY price ASC
        LIMIT  ?
    """, (f"%{drug_keyword}%", matched_name, max_results * 4)).fetchall()

    # Deduplicate by name
    seen = set()
    alternatives = []
    for r in rows:
        key = r['name'].lower().strip()
        if key not in seen:
            seen.add(key)
            alternatives.append({
                "name"        : r['name'],
                "price"       : round(r['price'], 2),
                "manufacturer": r['manufacturer'] or "Unknown",
                "salt"        : r['salt'],
                "source"      : r['source'],
            })
        if len(alternatives) >= max_results:
            break

    return {
        "success"      : True,
        "input_name"   : medicine_name,
        "matched_name" : matched_name,
        "match_score"  : round(score, 1),
        "salt"         : salt,
        "drug_keyword" : drug_keyword,
        "alternatives" : alternatives,
        "error"        : "" if alternatives else "No cheaper alternatives found.",
    }