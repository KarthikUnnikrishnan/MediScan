"""
scanner/ml/extractor.py — Structured OCR extraction and context-aware normalization.

Parses raw OCR output into clean, structured components:
- Brand name candidate
- Active ingredients and strengths
- Clean manufacturer name
- Dosage form
- Normalized OCR text
"""

import re
import logging
from typing import Dict, Any, List, Optional
from .salt_parser import parse_salt

logger = logging.getLogger(__name__)

# Standard dosage forms
DOSAGE_FORMS = [
    "Tablet", "Capsule", "Syrup", "Suspension", "Injection",
    "Ointment", "Gel", "Cream", "Drops", "Solution", "Infusion",
    "Lotion", "Inhaler", "Respules", "Spray", "Patch"
]

# Manufacturer indicator tokens
_MFR_INDICATORS = re.compile(
    r'\b(?:mfd\s*by|manufactured\s*by|mfg\s*by|marketed\s*by|made\s*by)\b[:\s]*',
    re.IGNORECASE
)

_PHARMA_CORP_SUFFIXES = re.compile(
    r'\b(?:pharma(?:ceuticals)?|laboratories|lab|biotech|lifesciences?|healthcare|remedies|drugs)\b.*?\b(?:ltd|limited|pvt|inc)\b'
    r'|\b(?:pharma(?:ceuticals)?|laboratories|lab|biotech|lifesciences?|healthcare|remedies|drugs)\b'
    r'|\b(?:ltd|limited|pvt|inc)\b',
    re.IGNORECASE
)

# Text indicators for junk/metadata lines to skip when seeking brand or manufacturer
_JUNK_LINE_PATTERNS = re.compile(
    r'\b(?:batch|mfg|exp|date|mrp|rs|lic|no|keep|out|reach|children|store|cool|dry|place|schedule|h|warning|caution|dosage|direction|use|as|directed|physician|composition|contains|each|uncoated|film|coated)\b',
    re.IGNORECASE
)

# Small targeted OCR correction mapping for exceptional OCR confusion
_COMMON_OCR_WORD_CORRECTIONS = {
    'parpcetaiiol': 'paracetamol',
    'parcetamol': 'paracetamol',
    'paracetamoi': 'paracetamol',
    'diclofenrc': 'diclofenac',
    'diclofenac': 'diclofenac',
    'azithral5oo': 'azithral 500',
    'azithral50o': 'azithral 500',
    'azithral': 'azithral',
    'dolo65o': 'dolo 650',
    'dolo6so': 'dolo 650',
    'dolo': 'dolo',
    'augmentin625': 'augmentin 625',
    'augmentin': 'augmentin',
    'crocin': 'crocin',
}


def normalize_ocr_text(raw_text: str) -> str:
    """
    Intelligently repair common OCR errors using context clues.
    Does NOT perform blind global character substitutions.
    """
    if not raw_text:
        return ""

    text = raw_text.strip()

    # 1. Repair digits inside dosage/number contexts
    # e.g., "65O mg" -> "650 mg", "5OO mg" -> "500 mg", "l25 mg" -> "125 mg", "S0 mg" -> "50 mg"
    def _fix_dosage_digits(match):
        num_str = match.group(1)
        unit_str = match.group(2)
        # Contextual digit substitution inside numbers next to units
        fixed_num = (
            num_str.replace('O', '0')
                   .replace('o', '0')
                   .replace('I', '1')
                   .replace('l', '1')
                   .replace('S', '5')
                   .replace('s', '5')
                   .replace('B', '8')
        )
        return f"{fixed_num}{unit_str}"

    text = re.sub(
        r'\b([0-9OolISsB]+)\s*(mg|ml|mcg|g|iu|%)\b',
        _fix_dosage_digits,
        text,
        flags=re.IGNORECASE
    )

    # 2. Repair brand/number combinations like "DOLO65O" -> "DOLO 650"
    # MUST contain at least one actual digit to prevent splitting regular words like "Paracetamol" or "Pharmaceuticals"
    def _split_and_fix_brand_strength(match):
        brand_part = match.group(1)
        num_part = match.group(2)
        if not re.search(r'\d', num_part):
            return match.group(0)
        fixed_num = (
            num_part.replace('O', '0')
                    .replace('o', '0')
                    .replace('I', '1')
                    .replace('l', '1')
                    .replace('S', '5')
                    .replace('s', '5')
                    .replace('B', '8')
        )
        return f"{brand_part} {fixed_num}"

    text = re.sub(
        r'\b([A-Za-z]{2,})(?=[A-Za-z0-9]*\d)([0-9OolISsB]{2,5})\b',
        _split_and_fix_brand_strength,
        text
    )

    # 3. Contextual word repairs for known pharmaceutical OCR corruptions
    lines = text.splitlines()
    repaired_lines = []
    for line in lines:
        tokens = line.split()
        repaired_tokens = []
        for token in tokens:
            clean_tok = re.sub(r'[^a-zA-Z0-9]', '', token).lower()
            if clean_tok in _COMMON_OCR_WORD_CORRECTIONS:
                # Replace token while preserving casing
                replacement = _COMMON_OCR_WORD_CORRECTIONS[clean_tok]
                repaired_tokens.append(replacement)
            else:
                repaired_tokens.append(token)
        repaired_line = " ".join(repaired_tokens).strip()
        if repaired_line:
            repaired_lines.append(repaired_line)

    normalized = "\n".join(repaired_lines).strip()
    return normalized


DOSAGE_FORM_ABBREVIATIONS = [
    (re.compile(r'\b(?:syp|syr)\b', re.IGNORECASE), "Syrup"),
    (re.compile(r'\b(?:susp)\b', re.IGNORECASE), "Suspension"),
    (re.compile(r'\b(?:tab|tabs)\b', re.IGNORECASE), "Tablet"),
    (re.compile(r'\b(?:cap|caps)\b', re.IGNORECASE), "Capsule"),
    (re.compile(r'\b(?:inj)\b', re.IGNORECASE), "Injection"),
    (re.compile(r'\b(?:oint)\b', re.IGNORECASE), "Ointment"),
    (re.compile(r'\b(?:gtt|drops?)\b', re.IGNORECASE), "Drops"),
    (re.compile(r'\b(?:soln?|sol)\b', re.IGNORECASE), "Solution"),
    (re.compile(r'\b(?:inh|inhaler)\b', re.IGNORECASE), "Inhaler"),
    (re.compile(r'\b(?:resp)\b', re.IGNORECASE), "Respules"),
]


def extract_dosage_form(text: str) -> Optional[str]:
    """
    Detect pharmaceutical dosage form (Tablet, Capsule, Syrup, etc.).
    Supports full names and standard medical abbreviations (Syp, Tab, Cap, etc.).
    """
    for form in DOSAGE_FORMS:
        # Match as distinct word
        if re.search(r'\b' + re.escape(form) + r'(?:s|\b)', text, re.IGNORECASE):
            return form

    for pattern, form in DOSAGE_FORM_ABBREVIATIONS:
        if pattern.search(text):
            return form

    return None


def extract_manufacturer(raw_text: str) -> Optional[str]:
    """
    Extract clean manufacturer name from OCR.
    Does NOT return the entire paragraph or address lines.
    """
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    # Check for explicit "Mfg by" or "Manufactured by" indicator
    for line in lines:
        m = _MFR_INDICATORS.search(line)
        if m:
            sub = line[m.end():].strip()
            # Stop before address / license tokens
            sub = re.split(r'[,;\n]|(?:\bat\b|\bplot\b|\bphase\b|\bindustrial\b)', sub, flags=re.IGNORECASE)[0].strip()
            if len(sub) > 2 and not _JUNK_LINE_PATTERNS.search(sub):
                return sub

    # Look for line with pharma corporate suffix (e.g. "Alembic Pharmaceuticals Ltd")
    for line in lines:
        if _JUNK_LINE_PATTERNS.search(line):
            continue
        if _PHARMA_CORP_SUFFIXES.search(line):
            clean = re.sub(r'^[^a-zA-Z0-9]+', '', line).strip()
            # Truncate after company ending (Ltd, Inc, etc.)
            end_match = re.search(r'\b(?:ltd|limited|pvt|inc)\b', clean, re.IGNORECASE)
            if end_match:
                clean = clean[:end_match.end()].strip()
            if 3 < len(clean) < 60:
                return clean

    return None


def extract_prominent_trade_name(text: str, manufacturer: Optional[str] = None, generic: Optional[str] = None) -> Optional[str]:
    """
    Extract high-confidence trade/brand name candidates (e.g. "P-650", "1-AL", "Dolo 650").
    Differentiates trade names from manufacturer names, generic salts, and dosage forms.
    """
    if not text:
        return None

    generic_words = {
        'paracetamol', 'levocetirizine', 'ibuprofen', 'amoxicillin', 'azithromycin',
        'pantoprazole', 'cetirizine', 'metformin', 'atorvastatin', 'omeprazole',
        'tablets', 'tablet', 'capsule', 'capsules', 'syrup', 'suspension', 'injection',
        'drops', 'drop', 'ip', 'bp', 'usp', 'each', 'contains', 'uncoated', 'coated',
        'excipients', 'dosage', 'store', 'warning', 'schedule'
    }
    mfr_words = set(re.findall(r'[a-zA-Z]+', (manufacturer or '').lower()))

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # 1. Alphanumeric / hyphenated trade codes (e.g. P-650, 1-AL, PCM-650, Pan-D, Meftal-P, D-650)
    for line in lines:
        for match in re.finditer(r'\b([A-Za-z0-9]{1,6}-[A-Za-z0-9]{1,6}(?:\s*(?:Tablets?|Capsules?|Suspension))?)\b', line, re.I):
            cand = match.group(1).strip()
            if not re.match(r'^(?:mg|ml|mcg|kg|iu|ph|lic|no|lot|exp|mfg)$', cand, re.I):
                return cand

    # 2. Prominent Brand + Strength (e.g. Dolo 650, Calpol 500, Crocin 650, Augmentin 625)
    for line in lines:
        for match in re.finditer(r'\b([A-Za-z]{3,15}\s+\d{2,4}(?:\s*(?:Tablets?|Capsules?))?)\b', line, re.I):
            cand = match.group(1).strip()
            first_w = cand.split()[0].lower()
            if first_w not in generic_words and first_w not in mfr_words:
                return cand

    # 3. Prominent Brand before dosage form (e.g. "Crocin Advance Tablet", "Augmentin Duo Tablet", "Calpol Tablet")
    for line in lines:
        m = re.search(r'^\s*([A-Za-z0-9\s-]{2,25})\s+(?:Tablets?|Capsules?|Suspension|Syrup)\b', line, re.I)
        if m:
            cand = m.group(1).strip()
            words = [w.lower() for w in cand.split()]
            if not any(w in generic_words for w in words) and not any(w in mfr_words for w in words):
                if 1 <= len(words) <= 3:
                    return cand

    return None


def extract_brand_candidate(text: str) -> Optional[str]:
    """
    Heuristically extract the most plausible brand name token from OCR.
    """
    # Exclude common non-brand and pure generic active salt words
    exclude_words = {
        'tab', 'tabs', 'tablet', 'tablets', 'cap', 'caps', 'capsule', 'capsules',
        'syp', 'syrup', 'suspension', 'inj', 'injection', 'ointment', 'gel',
        'cream', 'drops', 'drop', 'ip', 'bp', 'usp', 'rx', 'dr', 'od', 'bd', 'tds', 'hs', 'sos',
        'mg', 'ml', 'mcg', 'g', 'for', 'the', 'use', 'store', 'reach',
        'composition', 'each', 'contains', 'film', 'coated', 'batch', 'uncoated',
        'paracetamol', 'levocetirizine', 'ibuprofen', 'amoxicillin', 'azithromycin',
        'pantoprazole', 'cetirizine', 'metformin', 'atorvastatin', 'omeprazole',
        'aceclofenac', 'diclofenac', 'calcium', 'calcitriol', 'vitamin', 'vit',
        'd3', 'aspirin', 'clopidogrel', 'telmisartan', 'amlodipine', 'salbutamol',
        'ciprofloxacin', 'ofloxacin', 'metronidazole', 'tinidazole',
    }

    words = text.split()
    for i, w in enumerate(words):
        clean_w = re.sub(r'[^a-zA-Z0-9-]', '', w)
        if len(clean_w) < 2 or clean_w.lower() in exclude_words or clean_w.isdigit():
            continue
        # Skip pure strength values like 500mg, 650mg, 10ml, etc.
        if re.match(r'^\d+(?:\.\d+)?(?:mg|ml|mcg|g|iu|%|tabs?|caps?)?$', clean_w, re.IGNORECASE):
            continue
        # Check if immediately following word is a strength number (e.g. "650", "625", "500", "250/5")
        if i + 1 < len(words):
            num_match = re.search(r'(\d+(?:\.\d+)?)', words[i + 1])
            if num_match:
                next_clean = num_match.group(1)
                if 1 <= len(next_clean) <= 4:
                    return f"{clean_w} {next_clean}"
        return clean_w

    return None


def extract_structured_ocr(raw_ocr: str) -> Dict[str, Any]:
    """
    Main extraction interface.
    Parses OCR text into a structured dictionary.

    Returns
    -------
    dict:
        {
            "brand": str or None,
            "detected_label_name": str or None,
            "ingredients": list of dict,
            "manufacturer": str or None,
            "dosage_form": str or None,
            "raw_ocr": str,
            "normalized_ocr": str
        }
    """
    normalized = normalize_ocr_text(raw_ocr)
    dosage_form = extract_dosage_form(normalized) or extract_dosage_form(raw_ocr)
    manufacturer = extract_manufacturer(raw_ocr)

    trade_name = (
        extract_prominent_trade_name(raw_ocr, manufacturer=manufacturer) or
        extract_prominent_trade_name(normalized, manufacturer=manufacturer)
    )
    brand = trade_name or extract_brand_candidate(normalized)
    detected_label_name = trade_name or brand

    # If brand was found, strip it from OCR text before parsing active salts
    # so brand codes (e.g. 'Dolo-650', '1-AL') do not contaminate active chemical ingredient names
    salt_search_text = normalized
    if brand:
        brand_pattern = re.compile(r'\b' + re.escape(brand) + r'(?:\s*(?:Tablets?|Capsules?|Suspension))?\b', re.IGNORECASE)
        salt_search_text = brand_pattern.sub(' ', normalized).strip()
        brand_base = brand.split()[0].strip()
        if len(brand_base) >= 3 and brand_base.lower() != "tablet":
            salt_search_text = re.sub(r'\b' + re.escape(brand_base) + r'\b', ' ', salt_search_text, flags=re.IGNORECASE).strip()

    # Attempt to extract ingredients from salt search text
    ingredients = parse_salt(salt_search_text)
    if not ingredients or not any(ing.get("canonical_name") for ing in ingredients):
        ingredients = parse_salt(normalized)

    # Check for explicit generic label line (e.g. from structured OCR or prompt format)
    generic = None
    gen_match = re.search(r'^\s*GENERIC:\s*(.+)$', raw_ocr, flags=re.MULTILINE | re.IGNORECASE)
    if gen_match:
        generic = gen_match.group(1).strip()

    return {
        "brand": brand,
        "generic": generic,
        "detected_label_name": detected_label_name,
        "ingredients": ingredients,
        "manufacturer": manufacturer,
        "dosage_form": dosage_form,
        "raw_ocr": raw_ocr,
        "normalized_ocr": normalized,
    }


def build_descriptive_medicine_name(
    verified_med: Optional[Dict[str, Any]] = None,
    extracted: Optional[Dict[str, Any]] = None,
    generic_text: Optional[str] = None
) -> str:
    """
    Construct a clean, descriptive clinical medicine name (e.g., 'Levocetirizine Tablets 5mg')
    instead of an isolated brand abbreviation (e.g., '1-AL').
    Applies across all input types: package scans, generic strips, branded drugs, and prescriptions.
    """
    verified_med = verified_med or {}
    extracted = extracted or {}

    # 1. If explicit generic label text is available from OCR/vision (e.g. 'LEVOCETIRIZINE TABLETS IP 5 mg')
    raw_generic = (
        generic_text or
        extracted.get("generic") or
        extracted.get("generic_name") or
        ""
    ).strip()

    if raw_generic and raw_generic.lower() not in ("none", "null", "blank", "n/a", "generic"):
        # Strip pharmacopeial tokens: IP, BP, USP with or without dots
        cleaned = re.sub(r'\b(?:I\.?P\.?|B\.?P\.?|U\.?S\.?P\.?)\b', '', raw_generic, flags=re.IGNORECASE)
        # Collapse whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        # Normalize unit spacing: "5 mg" -> "5mg"
        cleaned = re.sub(r'(\d+(?:\.\d+)?)\s*(mg|ml|mcg|g|iu|%)\b', r'\1\2', cleaned, flags=re.IGNORECASE)

        words = cleaned.split()
        title_words = []
        for w in words:
            if re.match(r'^\d+(?:\.\d+)?(?:mg|ml|mcg|g|iu|%)$', w, re.IGNORECASE):
                title_words.append(w.lower())
            elif w.lower() in ('and', '&', 'with', 'for', 'in', 'of'):
                title_words.append(w.lower())
            else:
                title_words.append(w.capitalize())
        cand_name = " ".join(title_words)
        if len(cand_name) >= 4 and re.search(r'[a-zA-Z]{3,}', cand_name):
            return cand_name

    # 2. Determine dosage form
    dosage_form = (
        verified_med.get("dosage_form") or
        extracted.get("dosage_form") or
        "Tablet"
    ).strip()

    form_lower = dosage_form.lower()
    if form_lower in ("tablet", "tablets"):
        form_str = "Tablets"
    elif form_lower in ("capsule", "capsules"):
        form_str = "Capsules"
    elif form_lower in ("drops", "drop"):
        form_str = "Drops"
    elif form_lower in ("syrup", "suspension", "injection", "ointment", "gel", "cream", "solution", "inhaler", "spray", "infusion", "lotion", "respules"):
        form_str = dosage_form.capitalize()
    else:
        form_str = dosage_form.capitalize()

    # 3. Construct from parsed salt composition
    parsed_salts: List[Dict[str, Any]] = (
        verified_med.get("parsed_salt") or
        extracted.get("ingredients") or
        []
    )

    if not parsed_salts and verified_med.get("salt"):
        parsed_salts = parse_salt(verified_med.get("salt", ""))

    if parsed_salts:
        salt_names = []
        strengths = []
        for s in parsed_salts:
            ing = s.get("canonical_name") or s.get("ingredient") or ""
            ing = ing.strip().title()
            if ing and ing not in salt_names:
                salt_names.append(ing)

            st = s.get("strength")
            u = (s.get("unit") or "mg").lower()
            if st is not None:
                st_str = str(st).strip()
                if any(unit_tok in st_str.lower() for unit_tok in ("mg", "ml", "mcg", "g", "iu", "%")):
                    strengths.append(st_str.lower())
                else:
                    strengths.append(f"{st_str}{u}")

        if salt_names:
            combined_salts = " & ".join(salt_names)
            strength_str = "/".join(strengths) if strengths else ""
            if strength_str:
                return f"{combined_salts} {form_str} {strength_str}".strip()
            return f"{combined_salts} {form_str}".strip()

    # 4. Fallback to verified_med name if available, else label candidate
    return verified_med.get("name") or extracted.get("detected_label_name") or extracted.get("brand") or "Medication"

