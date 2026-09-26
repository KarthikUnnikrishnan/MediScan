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
    tokens = text.split()
    repaired_tokens = []
    for token in tokens:
        clean_tok = re.sub(r'[^a-zA-Z0-9]', '', token).lower()
        if clean_tok in _COMMON_OCR_WORD_CORRECTIONS:
            # Replace token while preserving casing
            replacement = _COMMON_OCR_WORD_CORRECTIONS[clean_tok]
            repaired_tokens.append(replacement)
        else:
            repaired_tokens.append(token)

    normalized = " ".join(repaired_tokens)
    # Normalize excess whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def extract_dosage_form(text: str) -> Optional[str]:
    """
    Detect pharmaceutical dosage form (Tablet, Capsule, Syrup, etc.).
    """
    for form in DOSAGE_FORMS:
        # Match as distinct word
        if re.search(r'\b' + re.escape(form) + r'(?:s|\b)', text, re.IGNORECASE):
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


def extract_brand_candidate(text: str) -> Optional[str]:
    """
    Heuristically extract the most plausible brand name token from OCR.
    """
    # Exclude common non-brand words
    exclude_words = {
        'tablet', 'tablets', 'capsule', 'capsules', 'syrup', 'suspension',
        'injection', 'ointment', 'gel', 'cream', 'drops', 'ip', 'bp', 'usp',
        'mg', 'ml', 'mcg', 'g', 'for', 'the', 'use', 'store', 'reach',
        'composition', 'each', 'contains', 'film', 'coated', 'batch'
    }

    words = text.split()
    candidates = []
    for w in words:
        clean_w = re.sub(r'[^a-zA-Z0-9-]', '', w)
        if len(clean_w) >= 2 and clean_w.lower() not in exclude_words and not clean_w.isdigit():
            candidates.append(clean_w)

    if candidates:
        # Return first 1 or 2 words if they form a brand (e.g. "Dolo 650", "Augmentin 625", "1-AL")
        if len(candidates) >= 2 and re.match(r'^\d+$', candidates[1]):
            return f"{candidates[0]} {candidates[1]}"
        return candidates[0]

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
    brand = extract_brand_candidate(normalized)

    # Attempt to extract ingredients from normalized text
    ingredients = parse_salt(normalized)

    return {
        "brand": brand,
        "ingredients": ingredients,
        "manufacturer": manufacturer,
        "dosage_form": dosage_form,
        "raw_ocr": raw_ocr,
        "normalized_ocr": normalized,
    }
