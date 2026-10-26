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
    'clavulanic': 'clavulanic acid',
    'clavulanate': 'clavulanic acid',
    'potassium clavulanate': 'clavulanic acid',
    'betamethasone dipropionate': 'betamethasone',
    'betamethasone valerate': 'betamethasone',
}

_KNOWN_ACTIVE_NAMES = [
    'paracetamol', 'acetaminophen', 'diclofenac', 'aceclofenac', 'ibuprofen',
    'calcium', 'calcitriol', 'vitamin d3', 'cholecalciferol', 'vitamin c',
    'ascorbic acid', 'vitamin b12', 'methylcobalamin', 'cyanocobalamin',
    'folic acid', 'pantoprazole', 'rabeprazole', 'omeprazole', 'esomeprazole',
    'levocetirizine', 'cetirizine', 'fexofenadine', 'loratadine', 'desloratadine',
    'amoxicillin', 'amoxycillin', 'clavulanic acid', 'clavulanic', 'clavulanate', 'azithromycin',
    'ciprofloxacin', 'ofloxacin', 'metronidazole', 'tinidazole',
    'metformin', 'glimepiride', 'gliclazide', 'vildagliptin', 'sitagliptin',
    'atorvastatin', 'rosuvastatin', 'simvastatin', 'aspirin', 'clopidogrel',
    'telmisartan', 'losartan', 'amlodipine', 'atenolol', 'metoprolol',
    'salbutamol', 'budesonide', 'formoterol', 'montelukast', 'theophylline',
    'dextromethorphan', 'chlorpheniramine', 'phenylephrine', 'ambroxol',
    'domperidone', 'ondansetron', 'ranitidine', 'famotidine', 'dicyclomine',
    'mefenamic acid', 'betamethasone', 'povidone iodine', 'povidone', 'iodine',
    'celecoxib', 'miconazole', 'fusidic acid', 'fusidic', 'gentamicin',
    'lactulose', 'terbinafine', 'acyclovir', 'mupirocin', 'prednisolone',
    'chlorhexidine', 'picosulfate', 'flavonoid', 'hesperidin', 'diosmin',
    'nitroglycerin', 'glyceryl trinitrate',
]

# Regex to capture: [Ingredient Name] ( [Strength] [Unit] )
# Units of measure for pharmaceutical active ingredients
_UNIT_PATTERN = r'(?:mg|ml|mcg|µg|g|gm|iu|units?|spores?|%|%\s*w/[vw]|w/v|w/w)(?:\s*/\s*\d*\s*(?:mg|ml|mcg|µg|g|gm|dose|puff))?'

# Regex to capture: [Ingredient Name] ( [Strength] [Unit] )
_SALT_BLOCK_RE = re.compile(
    rf'(?:^|\+|\band\b|(?<=\))|[,\n])\s*(.*?)\s*\(\s*([\d\.]+)\s*({_UNIT_PATTERN})\s*\)',
    re.IGNORECASE
)

# Regex to capture: [Ingredient Name] [Strength] [Unit] without parentheses
# e.g. "Fexofenadine Hydrochloride 120mg", "Paracetamol 650 mg"
_SALT_BLOCK_NO_PARENS_RE = re.compile(
    rf'(?:^|\+|\band\b|[,\n])\s*(.*?)\s+([\d\.]+)\s*({_UNIT_PATTERN})(?=\s*(?:\+|\band\b|[,\n]|$))',
    re.IGNORECASE
)

# Regex to capture prefix concentration: [Strength] [Unit] [Ingredient Name]
# e.g. "5% Dextrose", "0.5% Timolol", "1% Clotrimazole"
_PREFIX_CONCENTRATION_RE = re.compile(
    rf'(?:^|\+|\band\b|[,\n])\s*(?:.*?\b(?:inj|tab|cap|syp|susp)\b[\.\s]*)?\s*([\d\.]+)\s*(%|%\s*w/[vw]|w/v|w/w)\s+([a-zA-Z\s]+?)(?=\s*(?:\+|\band\b|[,\n]|$))',
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
    
    # Remove scraped junk prefixes (including leading dots, commas, ltd., etc.)
    clean = re.sub(r"^[\s\.\,\'\-\&]+", "", clean)
    clean = re.sub(
        r"^.*?\b(?:ltd|limited|pvt|laboratories|biotech|lifescience|antibiotics|ayurveda)[\s\.\,]*",
        "",
        clean,
        flags=re.IGNORECASE
    )
    clean = re.sub(r"^[\s\.\,\'\-\&]+", "", clean)

    # Strip sub-brand marketing modifiers like "advance", "plus", "forte", etc. when prefixed to ingredient
    clean = re.sub(
        r'^\s*(?:advance|plus|forte|duo|ultra|fast|relief|extra|pro|max|cr|sr|er|xr|ds|dt|od|mr)\s+',
        '',
        clean,
        flags=re.IGNORECASE
    )

    # Strip dosage form words and pharmacopoeial designations without discarding preceding ingredient text
    clean = re.sub(
        r'\b(?:syp|syr|tab|tabs|cap|caps|inj|susp|oint|tablet|tablets|capsule|capsules|syrup|suspension|injection|drops|drop|ointment|gel|cream|solution|duo|forte)\b',
        ' ',
        clean,
        flags=re.IGNORECASE
    )
    clean = re.sub(r'\b(?:ip|bp|usp|i\.p\.|b\.p\.|u\.s\.p\.)\b', ' ', clean, flags=re.IGNORECASE)
    
    # Strip salt modifiers, but preserve the name if it is an active salt like calcium
    mod_stripped = _SALT_MODIFIERS.sub("", clean)
    mod_stripped = re.sub(r"[^a-z0-9\s]", " ", mod_stripped).strip()
    if mod_stripped:
        clean = mod_stripped
    else:
        clean = re.sub(r"[^a-z0-9\s]", " ", clean).strip()
    
    clean = re.sub(r"\s+", " ", clean).strip()

    # Check if a known active chemical name is present in clean (e.g. "advance paracetamol" -> "paracetamol")
    for act in _KNOWN_ACTIVE_NAMES:
        if re.search(r'\b' + re.escape(act) + r'\b', clean):
            clean = act
            break
    
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
    clean = re.sub(r"^[\s\.\,\'\-\&]+", "", clean)
    clean = re.sub(
        r"^.*?\b(?:ltd|limited|pvt|laboratories|biotech|lifescience|antibiotics|ayurveda)[\s\.\,]*",
        "",
        clean,
        flags=re.IGNORECASE
    )
    clean = re.sub(r"^[\s\.\,\'\-\&]+", "", clean)

    # Strip sub-brand marketing modifiers like "advance", "plus", "forte", etc. when prefixed to ingredient
    clean = re.sub(
        r'^\s*(?:advance|plus|forte|duo|ultra|fast|relief|extra|pro|max|cr|sr|er|xr|ds|dt|od|mr)\s+',
        '',
        clean,
        flags=re.IGNORECASE
    )

    # Strip dosage form words and pharmacopoeial designations from the ingredient
    clean = re.sub(
        r'\b(?:tablet|tablets|capsule|capsules|syrup|suspension|injection|drops|drop|ointment|gel|cream|solution|duo|forte)\b',
        ' ',
        clean,
        flags=re.IGNORECASE
    )
    clean = re.sub(r'\b(?:ip|bp|usp|i\.p\.|b\.p\.|u\.s\.p\.)\b', ' ', clean, flags=re.IGNORECASE)

    # Remove common scraped trailing words
    clean = re.sub(r"\badd$", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


_DOSAGE_LINE_RE = re.compile(
    r'\b(?:DOSE|DOSES|DAILY|MAXIMUM|MAX|EVERY|HOURS?|PER\s*DAY)\b',
    re.IGNORECASE
)

_DIRECT_STRENGTH_PATTERN = re.compile(
    r'\b(?:IP|BP|USP)?\s*(\d+(?:\.\d+)?)\s*(MG|MCG|UG|µG|G|GM|ML|%|IU|UNITS?)\b',
    re.IGNORECASE
)

_NON_BRAND_PREFIXES = {
    'IP', 'BP', 'USP', 'MG', 'ML', 'MCG', 'TAB', 'TABS', 'TABLET', 'TABLETS',
    'CAP', 'CAPS', 'CAPSULE', 'CAPSULES', 'NO', 'EXP', 'LOT', 'BATCH', 'FOR',
    'OF', 'IN', 'DOSE', 'DOSES', 'EVERY', 'DAILY', 'MAX'
}


def extract_strengths_from_text(text: str) -> set:
    """
    Extract canonical strength tokens e.g. {'180mg'}, {'5mg'}, {'0.5%'} from text.
    Handles '180 mg', '5mg', 'IP 180 mg', '0.5% w/v', '10 mcg', '500 iu', etc.
    Also handles hyphenated/brand numbers (e.g. FEXOMINE-180, P-650).
    Ignores numbers on lines containing dosage/frequency instructions (DOSE, DAILY, MAXIMUM, etc.).
    Prefers strengths next to active salt names (e.g. "Paracetamol IP 650 mg").
    If brand has a number ("Fexomine-180") and a strength on pack equals that number,
    trusts it and drops other OCR strengths.
    """
    if not text:
        return set()

    raw_lines = [l.strip() for l in re.split(r'[\r\n]+', text) if l.strip()]
    if len(raw_lines) <= 1:
        raw_lines = [l.strip() for l in re.split(r'[\r\n;]+', text) if l.strip()]

    all_strengths = set()
    salt_adjacent_strengths = set()
    brand_nums = set()

    for line in raw_lines:
        line_upper = line.upper()

        # Check for brand numbers e.g. "FEXOMINE-180", "P-650", "DOLO 650", "LORINOL 10"
        if not _DOSAGE_LINE_RE.search(line_upper):
            for m in re.finditer(r'\b([A-Za-z0-9]+)[- ](\d{1,4})\b', line_upper):
                word, num = m.group(1), m.group(2)
                if word not in _NON_BRAND_PREFIXES:
                    brand_nums.add(num)

        # 1. Ignore numbers on lines that contain dosage/frequency instructions
        if _DOSAGE_LINE_RE.search(line_upper):
            continue

        # Extract direct strengths from this line
        line_strengths = set()
        for match in _DIRECT_STRENGTH_PATTERN.finditer(line_upper):
            val = match.group(1)
            unit = match.group(2).lower()
            if unit in ('ug', 'µg'):
                unit = 'mcg'
            elif unit == 'gm':
                unit = 'g'
            elif unit == 'units':
                unit = 'unit'
            try:
                val_f = float(val)
                val_str = str(int(val_f)) if val_f.is_integer() else str(val_f)
            except ValueError:
                val_str = val
            st = f"{val_str}{unit}"
            line_strengths.add(st)
            all_strengths.add(st)

        # Prefer strength next to salt name
        line_lower = line.lower()
        has_salt = any(re.search(r'\b' + re.escape(salt) + r'\b', line_lower) for salt in _KNOWN_ACTIVE_NAMES)
        if has_salt and line_strengths:
            salt_adjacent_strengths.update(line_strengths)

    # Fallback hyphenated numbers if no direct unit strengths found
    if not all_strengths and brand_nums:
        dominant_unit = 'mg' if ('MG' in text.upper() or 'TABLET' in text.upper() or 'CAPSULE' in text.upper()) else None
        if dominant_unit:
            for b_num in brand_nums:
                all_strengths.add(f"{b_num}{dominant_unit}")

    # Prefer salt-adjacent strengths if any were found
    chosen_strengths = salt_adjacent_strengths if salt_adjacent_strengths else all_strengths

    # Bug 3: If brand has a number ("Fexomine-180") and a strength on pack equals that number,
    # trust it and drop other OCR strengths (e.g. 100mg typo).
    if brand_nums:
        matching_brand_strengths = {
            s for s in chosen_strengths
            if any(re.match(rf'^{b_num}[a-z%]+$', s) for b_num in brand_nums)
        }
        if matching_brand_strengths:
            chosen_strengths = matching_brand_strengths

    return chosen_strengths


def extract_active_salt_set(salt_str: Optional[str]) -> set:
    """
    Extract set of normalized canonical active ingredient names from a salt string.
    e.g. 'levocetirizine (5mg) + montelukast (10mg)' -> {'levocetirizine', 'montelukast'}
    """
    if not salt_str:
        return set()
    # Filter out lines containing dosage/frequency instructions
    clean_lines = [l for l in salt_str.splitlines() if not _DOSAGE_LINE_RE.search(l)]
    salt_text = "\n".join(clean_lines) if clean_lines else salt_str
    parsed = parse_salt(salt_text)
    salts = set()
    for item in parsed:
        cname = item.get("canonical_name")
        if cname and not _DOSAGE_LINE_RE.search(cname):
            salts.add(cname)
    if not salts:
        clean = salt_text.lower()
        for act in _KNOWN_ACTIVE_NAMES:
            if re.search(r'\b' + re.escape(act) + r'\b', clean):
                salts.add(act)
    return salts


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
    is_prefix = False
    if not matches:
        matches = list(_SALT_BLOCK_NO_PARENS_RE.finditer(s))
    if not matches:
        matches = list(_PREFIX_CONCENTRATION_RE.finditer(s))
        is_prefix = bool(matches)
    ingredients: List[Dict[str, Any]] = []

    for m in matches:
        if is_prefix:
            raw_strength = m.group(1).strip()
            raw_unit = m.group(2).strip()
            raw_name = m.group(3).strip()
        else:
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

    # Unquantified active ingredients fallback: detect named chemical actives even without explicit dosage numbers
    if not ingredients:
        s_lower = s.lower()
        found_canons = set()
        for act in _KNOWN_ACTIVE_NAMES:
            if re.search(r'\b' + re.escape(act) + r'\b', s_lower):
                canon = normalize_ingredient_name(act)
                if canon and canon not in found_canons:
                    found_canons.add(canon)
                    ingredients.append({
                        "ingredient": act,
                        "canonical_name": canon,
                        "strength": None,
                        "unit": "mg",
                        "raw": s,
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
