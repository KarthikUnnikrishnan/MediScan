"""
Model 3 — Generic Medicine Finder
Given a medicine name → find same-salt cheaper alternatives
Works for any medicine: strips, prescriptions, brand names, generic names
"""

import os, re, sqlite3
from rapidfuzz import process, fuzz

DB_PATH = r"D:\Coding Section\Mediscan\db_cache\medicines.sqlite"

_conn      = None
_all_names = []


def load_db():
    global _conn, _all_names
    _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    rows = _conn.execute(
        "SELECT DISTINCT name FROM medicines WHERE name IS NOT NULL"
    ).fetchall()
    _all_names = [r[0] for r in rows if r[0]]
    print(f"[GenericFinder] DB loaded — {len(_all_names):,} medicine names")


# ── Dose pattern: 5mg, 650 mg, 5%, 500ml, 1.5g, 10mcg, 5 D 5% etc. ──────
_DOSE_PATTERN = re.compile(
    r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|ug|ml|g|kg|iu|units?|%|w/v|w/w|v/v)\b'
    r'|(?<!\w)\d+(?:\.\d+)?\s*(?:mg|mcg|ug|ml|g|kg|iu|%)',
    re.IGNORECASE
)

# ── Standalone bare numbers (e.g. "650" in "Dolo 650") — strip only if
#    followed by end-of-string or a noise word, NOT if preceded by hyphen/letter
_BARE_NUMBER = re.compile(
    r'(?<![A-Za-z\-])\b\d+(?:\.\d+)?\b(?![A-Za-z%])',
)

# ── Pharmaceutical form words ─────────────────────────────────────────────
_FORM_WORDS = re.compile(
    r'\b('
    r'tablets?|tab\.?|capsules?|cap\.?|caps\.?|'
    r'injection|inj\.?|syrup|syp\.?|cream|gel|'
    r'drops?|solution|suspension|infusion|ointment|'
    r'lotion|spray|inhaler|patch|sachet|powder'
    r')\b',
    re.IGNORECASE
)

# ── Regulatory / release / qualifier words ────────────────────────────────
_REG_WORDS = re.compile(
    r'\b('
    r'ip|bp|usp|ep|'                          # regulatory
    r'sr|cr|xr|er|mr|la|od|'                  # release types
    r'forte|plus|ds|junior|kid|paediatric|pediatric|'
    r'new|extra|ultra|max|mini|rapid|fast|retard'
    r')\b',
    re.IGNORECASE
)

# ── Composite phrases (multi-word noise) ──────────────────────────────────
_COMPOSITE_NOISE = re.compile(
    r'\b(film[\s\-]?coated|extended[\s\-]?release|'
    r'sustained[\s\-]?release|modified[\s\-]?release)\b',
    re.IGNORECASE
)

# ── Route-of-administration words ─────────────────────────────────────────
_ROUTE_WORDS = re.compile(
    r'\b(oral|topical|intravenous|iv\b|im\b|sc\b|intramuscular|'
    r'subcutaneous|nasal|ophthalmic|otic|rectal|vaginal|'
    r'transdermal|sublingual)\b',
    re.IGNORECASE
)


def _clean_for_matching(name: str) -> str:
    """
    Remove dose, form, route, and regulatory words from an OCR medicine
    string so fuzzy matching targets only the drug name itself.

    Cleaning order matters:
      1. Composite multi-word phrases (film coated, extended release, ...)
      2. Dose patterns with units (5mg, 650 mg, 5%, ...)
      3. Form words (tablet, capsule, syrup, ...)
      4. Regulatory / release / qualifier single words (IP, BP, SR, ...)
      5. Route words (oral, IV, ...)
      6. Normalise whitespace

    Examples:
      'LEVOCETIRIZINE TABLETS IP 5 mg'     → 'LEVOCETIRIZINE'
      'Dolo 650 Tablet'                    → 'Dolo 650'
      'Augmentin 625mg Tablet'             → 'Augmentin'
      'Pan-D Capsule'                      → 'Pan-D'
      'Amoxicillin 500mg Capsule BP'       → 'Amoxicillin'
      'Metformin 500 SR Tablet'            → 'Metformin'
      'Pantoprazole 40 mg Tablet EC'       → 'Pantoprazole'
    """
    s = _COMPOSITE_NOISE.sub(' ', name)
    s = _DOSE_PATTERN.sub(' ', s)
    s = _FORM_WORDS.sub(' ', s)
    s = _REG_WORDS.sub(' ', s)
    s = _ROUTE_WORDS.sub(' ', s)
    s = ' '.join(s.split()).strip()
    return s


def _candidate_queries(medicine_name: str) -> list:
    """
    Build a ranked list of search strings to try, from most targeted
    to most general.  find_generics() stops at the first query that
    achieves a score >= 75.

    IMPORTANT: We never include the raw original name as a candidate
    when it contains noise (dose/form/regulatory words) because WRatio
    can wrongly match dose numbers to unrelated medicine names.

    For 'LEVOCETIRIZINE TABLETS IP 5 mg':
      1. 'LEVOCETIRIZINE'   ← full cleaned (1 word) → best
      2. 'LEVOCETIRIZINE'   ← first word (deduped)

    For 'Augmentin 625mg Tablet':
      1. 'Augmentin'        ← full cleaned (1 word)
      2. 'Augmentin'        ← first word (deduped)

    For 'Pan-D Capsule':
      1. 'Pan-D'            ← full cleaned
      2. 'Pan-D'            ← first word (deduped)

    For 'Metformin 500 SR Tablet':
      1. 'Metformin'        ← full cleaned (500 and SR stripped)

    For 'Dolo 650 Tablet':
      1. 'Dolo'             ← full cleaned (650 is a bare number — kept since
                              no unit suffix, and _REG_WORDS doesn't strip it)
         Actually: 'Dolo 650' if 650 has no unit → stays in cleaned
    """
    cleaned    = _clean_for_matching(medicine_name)
    words      = cleaned.split()
    orig_words = medicine_name.split()

    candidates = []

    # 1. Full cleaned string — primary target (noise removed)
    if cleaned:
        candidates.append(cleaned)

    # 2. First word of cleaned (core active ingredient / brand name)
    if words and len(words[0]) >= 3:
        candidates.append(words[0])

    # 3. First two words of cleaned
    if len(words) >= 2:
        candidates.append(' '.join(words[:2]))

    # 4. First two words of original ONLY if they don't contain dose noise
    #    (handles brand names like "Dolo 650" where the number is part of the brand)
    if len(orig_words) >= 2:
        two_orig = ' '.join(orig_words[:2])
        # Only add if it doesn't look like pure noise (e.g. avoid "5 mg")
        if not re.fullmatch(
            r'\d+(?:\.\d+)?\s*(?:mg|mcg|ml|g|kg|iu|%|w/v|w/w|v/v)?',
            orig_words[0], re.IGNORECASE
        ):
            candidates.append(two_orig)

    # NOTE: We deliberately do NOT add the full original (medicine_name) as
    # a fallback when it contains pharmaceutical noise words, because WRatio
    # fuzzy matching on "LEVOCETIRIZINE TABLETS IP 5 mg" would incorrectly
    # match "5 D 5% Infusion" with a high score due to the "5 mg" substring.
    # Only add original if it equals the cleaned version (no noise was stripped).
    if medicine_name.strip().lower() == cleaned.lower():
        candidates.append(medicine_name)

    # Remove duplicates (case-insensitive), preserve order, drop empties < 2 chars
    seen   = set()
    unique = []
    for c in candidates:
        key = c.strip().lower()
        if key and key not in seen and len(key) >= 2:
            seen.add(key)
            unique.append(c.strip())

    return unique


def find_generics(medicine_name: str, max_results: int = 5) -> dict:
    """
    Main function called from Django views.

    Args:
        medicine_name : raw name from OCR or prescription HTR
        max_results   : number of alternatives to return

    Returns dict with keys:
        success, input_name, matched_name, match_score,
        salt, drug_keyword, alternatives, error
    """
    if _conn is None:
        return {"success": False, "input_name": medicine_name,
                "alternatives": [], "error": "DB not loaded."}

    if not medicine_name or len(medicine_name.strip()) < 2:
        return {"success": False, "input_name": medicine_name,
                "alternatives": [], "error": "Medicine name too short."}

    cleaned_name = _clean_for_matching(medicine_name)
    candidates   = _candidate_queries(medicine_name)

    print(f"[GenericFinder] Input   : '{medicine_name}'")
    print(f"[GenericFinder] Cleaned : '{cleaned_name}'")
    print(f"[GenericFinder] Queries : {candidates}")

    # ── Step 1: Try multiple query candidates, pick best match ─────────
    # Strategy: run ALL queries, but stop early if we find a very good match.
    # Threshold = 75: high enough to require a real match on the cleaned name,
    # but low enough to accept partial brand-name hits.
    EARLY_EXIT_THRESHOLD = 75

    best_match = None
    best_score = 0
    best_query = None

    for query in candidates:
        result = process.extractOne(query, _all_names, scorer=fuzz.WRatio)
        if result:
            match_name, score, _ = result
            print(f"[GenericFinder]   query='{query}' → '{match_name}' score={score:.0f}")
            if score > best_score:
                best_match = match_name
                best_score = score
                best_query = query
            # Good enough match on a clean query — stop here
            if best_score >= EARLY_EXIT_THRESHOLD:
                break

    print(f"[GenericFinder] BEST    : query='{best_query}' matched='{best_match}' score={best_score:.0f}")

    if not best_match or best_score < 55:
        return {
            "success"     : False,
            "input_name"  : medicine_name,
            "alternatives": [],
            "error"       : f"No match found for '{medicine_name}' in database.",
        }

    # ── Step 2: Get salt of matched medicine ───────────────────────────
    row = _conn.execute(
        "SELECT salt FROM medicines WHERE name = ? LIMIT 1",
        (best_match,)
    ).fetchone()

    if not row or not row['salt'] or row['salt'].strip().lower() in ('nan', '', 'none'):
        return {
            "success"     : False,
            "input_name"  : medicine_name,
            "matched_name": best_match,
            "alternatives": [],
            "error"       : "Salt composition not found for this medicine.",
        }

    salt = row['salt'].strip().lower()

    # ── Step 3: Extract active ingredient keyword from salt ────────────
    # "levocetirizine hydrochloride (5mg)" → "levocetirizine hydrochloride"
    # "paracetamol (650mg) + ibuprofen (400mg)" → "paracetamol"
    # "(5% w/v)" → fallback to full salt
    drug_keyword = re.split(r'[\(\+\,]', salt)[0].strip()
    if len(drug_keyword) < 3:
        drug_keyword = salt

    # ── Step 4: SQL search for same-salt medicines ─────────────────────
    rows = _conn.execute("""
        SELECT name, price, manufacturer, source, salt
        FROM   medicines
        WHERE  LOWER(salt) LIKE ?
          AND  LOWER(name) != LOWER(?)
          AND  price IS NOT NULL
          AND  price > 0
        ORDER  BY price ASC
        LIMIT  ?
    """, (f"%{drug_keyword}%", best_match, max_results * 4)).fetchall()

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
        "matched_name" : best_match,
        "match_score"  : round(best_score, 1),
        "salt"         : salt,
        "drug_keyword" : drug_keyword,
        "alternatives" : alternatives,
        "error"        : "" if alternatives else "No cheaper alternatives found.",
    }