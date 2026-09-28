"""
scanner/ml/salt_parser.py — Robust chemical salt and active ingredient parser.

Converts free-text salt formulations (from medicines.sqlite or OCR) into
structured ingredient records with precise strengths and units.
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Common pharmaceutical salt modifiers that don't change the primary active moiety
_SALT_MODIFIERS = re.compile(
    r'\b(sodium|potassium|hydrochloride|dihydrochloride|maleate|succinate|'
    r'tartrate|fumarate|besylate|mesylate|phosphate|sulfate|sulphate|'
    r'citrate|calcium|hydrate|anhydrous|trihydrate|monohydrate|dihydrate|'
    r'valerate|propionate|dipropionate|acetate|gluconate|nitrate|lactate|'
    r'hcl|ip|bp|usp)\b',
    re.IGNORECASE
)

# Standard international INN <-> USAN synonyms
_SYNONYM_MAP = {
    'acetaminophen': 'paracetamol',
    'amoxicillin': 'amoxycillin',
    'levocetirizine': 'levocetirizine',
    'albuterol': 'salbutamol',
    'frusemide': 'furosemide',
    'cholecalciferol': 'vitamin d3',
    'pyridoxine': 'vitamin b6',
    'cyanocobalamin': 'vitamin b12',
    'methylcobalamin': 'vitamin b12',
    'folic acid': 'folic acid',
    'pantoprazole sodium': 'pantoprazole',
}

# Regex to capture: [Ingredient Name] ( [Strength] [Unit] )
# Supports:
#  - (500mg), (125 mg)
#  - (200mg/5ml), (120mg/5ml), (10mg/ml)
#  - (2% w/w), (1% w/v)
#  - (2.5million spores), (1000IU), (500mcg)
# Delimiters: +, and, comma, newline, or immediately following a preceding ')'
_SALT_BLOCK_RE = re.compile(
    r'(?:^|\+|\band\b|(?<=\))|[,\n])\s*(.*?)\s*\(\s*([\d\.]+)\s*([a-zA-Z%]+(?:\s*/\s*\d*\s*[a-zA-Z%]+|\s*[a-zA-Z%/\s]+)?)\s*\)',
    re.IGNORECASE
)

# Detect orphaned strength parenthesis at start or middle without ingredient
# e.g. "(250mg) + diclofenac (50mg)"
_ORPHANED_STRENGTH_RE = re.compile(
    r'(?:^|\+)\s*\(\s*[\d\.]+\s*[a-zA-Z%]+',
    re.IGNORECASE
)


def normalize_ingredient_name(name: str) -> str:
    """
    Produce a canonical base name for equivalence matching.
    Strips brand prefixes, dosage form words, salt modifiers (e.g. sodium, hydrochloride),
    punctuation, and extra whitespace.
    """
    if not name:
        return ""
    
    clean = name.lower().strip()
    
    # Remove scraped junk prefixes
    clean = re.sub(r"^['&][^a-zA-Z0-9]*", "", clean)
    clean = re.sub(
        r"^.*?\b(?:ltd|limited|pvt|laboratories|biotech|lifescience|antibiotics|ayurveda)\s*",
        "",
        clean,
        flags=re.IGNORECASE
    )

    # Strip leading dosage form words if OCR captured "Brand Tablet Ingredient"
    clean = re.sub(
        r'^(?:.*?\b(?:tablet|tablets|capsule|capsules|syrup|suspension|injection|drops|ointment|gel|cream|solution|duo|forte)\b\s*)',
        '',
        clean,
        flags=re.IGNORECASE
    )
    
    # Strip salt modifiers
    clean = _SALT_MODIFIERS.sub("", clean)
    
    # Strip non-alphanumeric except spaces
    clean = re.sub(r"[^a-z0-9\s]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    
    # If multiple words remain and the last word is a known chemical, take the chemical
    # Apply standard synonyms
    return _SYNONYM_MAP.get(clean, clean)


def clean_display_ingredient(name: str) -> str:
    """
    Clean the ingredient name for display without stripping legitimate
    chemical descriptors like 'sodium'.
    """
    if not name:
        return ""
    clean = name.strip()
    clean = re.sub(r"^['&][^a-zA-Z0-9]*", "", clean)
    clean = re.sub(
        r"^.*?\b(?:ltd|limited|pvt|laboratories|biotech|lifescience|antibiotics|ayurveda)\s*",
        "",
        clean,
        flags=re.IGNORECASE
    )

    # Strip leading dosage form words if OCR captured "Brand Tablet Ingredient"
    clean = re.sub(
        r'^(?:.*?\b(?:tablet|tablets|capsule|capsules|syrup|suspension|injection|drops|ointment|gel|cream|solution|duo|forte)\b\s*)',
        '',
        clean,
        flags=re.IGNORECASE
    )

    # Remove common scraped trailing words
    clean = re.sub(r"\badd$", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def parse_salt(salt_str: Optional[str]) -> List[Dict[str, Any]]:
    """
    Parse a free-text salt string into structured ingredient records.

    Parameters
    ----------
    salt_str : str or None
        e.g. 'amoxycillin (500mg) + clavulanic acid (125mg)'
        or 'diclofenac sodium (50mg) + paracetamol (325mg)'

    Returns
    -------
    list of dict
        [
            {
                "ingredient": str (display name, e.g. "diclofenac sodium"),
                "canonical_name": str (normalized base name, e.g. "diclofenac"),
                "strength": float or int (e.g. 50),
                "unit": str (e.g. "mg"),
                "raw": str
            }, ...
        ]
    """
    if not salt_str:
        return []

    s = salt_str.strip()
    if s.endswith('add'):
        s = s[:-3].strip()

    # If the salt string has orphaned parentheses like '(250mg) + ...',
    # check if the first chunk is missing an ingredient name
    if _ORPHANED_STRENGTH_RE.match(s):
        logger.debug("Corrupted salt string detected (orphaned strength): %r", salt_str)
        # Note: caller will know this has missing data if components don't match

    matches = list(_SALT_BLOCK_RE.finditer(s))
    ingredients: List[Dict[str, Any]] = []

    for m in matches:
        raw_name = m.group(1).strip()
        raw_strength = m.group(2).strip()
        raw_unit = m.group(3).strip()

        display_name = clean_display_ingredient(raw_name)
        canonical = normalize_ingredient_name(display_name)
        
        # If display name is empty (orphaned block), skip or mark invalid
        if not display_name and not canonical:
            continue

        # Clean unit: e.g. "mg / 5ml" -> "mg/5ml", "% w/w" -> "% w/w"
        unit_clean = re.sub(r'\s*/\s*', '/', raw_unit.lower())
        unit_clean = re.sub(r'\s+', ' ', unit_clean).strip()

        # Numeric strength conversion
        try:
            val = float(raw_strength)
            if val.is_integer():
                val = int(val)
        except ValueError:
            val = raw_strength

        ingredients.append({
            "ingredient": display_name.lower(),
            "canonical_name": canonical,
            "strength": val,
            "unit": unit_clean,
            "raw": m.group(0).strip(),
        })

    return ingredients


def is_corrupted_salt(salt_str: Optional[str]) -> bool:
    """
    Only reject genuinely corrupted salt strings:
    - Empty or whitespace-only
    - Starts with orphaned parenthesis like "(250mg) + ..."
      with NO ingredient name at all before the first (
    - Has zero parseable ingredients AND no recognisable
      ingredient word before any parenthesis
    """
    if not salt_str or not salt_str.strip():
        return True
    s = salt_str.strip()

    # Only corrupt if the ENTIRE string starts with a number+unit
    # in parens with nothing before it
    if re.match(r'^\(\s*[\d\.]+\s*[a-zA-Z%]+', s):
        return True

    # Must have at least one letter-word before any parenthesis
    before_paren = s.split('(')[0].strip()
    if not before_paren or not re.search(r'[a-zA-Z]{2,}', before_paren):
        return True

    return False


def compare_compositions(comp_a: List[Dict[str, Any]], comp_b: List[Dict[str, Any]], require_strength: bool = True) -> bool:
    """
    Strict comparison of two structured compositions.
    Returns True if and only if both compositions contain identical active ingredients
    and identical strengths (when require_strength=True).
    """
    if not comp_a or not comp_b:
        return False
    if len(comp_a) != len(comp_b):
        return False

    def _sort_key(ing):
        return ing.get("canonical_name", "")

    sorted_a = sorted(comp_a, key=_sort_key)
    sorted_b = sorted(comp_b, key=_sort_key)

    for ia, ib in zip(sorted_a, sorted_b):
        if ia.get("canonical_name") != ib.get("canonical_name"):
            return False
        if require_strength:
            sa = ia.get("strength")
            sb = ib.get("strength")
            ua = ia.get("unit")
            ub = ib.get("unit")
            if sa != sb or ua != ub:
                return False

    return True
