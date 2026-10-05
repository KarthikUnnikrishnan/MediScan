"""
scanner/ml/ai_validator.py

Three-tier AI validation for MediScan:
  Tier 1: Groq API (online)     — fast, accurate
  Tier 2: Ollama local (offline) — works without internet
  Tier 3: Rule-based fallback   — always works, no AI
"""

import os
import json
import logging
import re
import time
import socket
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

_GROQ_MODEL   = "llama-3.1-70b-versatile"
_OLLAMA_MODEL = "llama3.1:8b"
_OLLAMA_URL   = "http://localhost:11434/api/chat"

_groq_client      = None
_groq_available   = False
_ollama_available = False
_active_tier      = None


# ════════════════════════════════════════════════════════
# INIT
# ════════════════════════════════════════════════════════

def _check_internet():
    try:
        socket.setdefaulttimeout(2)
        socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        ).connect(("8.8.8.8", 53))
        return True
    except Exception:
        return False


def _check_ollama():
    try:
        import urllib.request
        resp = urllib.request.urlopen(
            "http://localhost:11434/api/tags", timeout=2
        )
        data = json.loads(resp.read())
        models = [
            m.get("name", "")
            for m in data.get("models", [])
        ]
        found = any(_OLLAMA_MODEL in m for m in models)
        if found:
            logger.info(
                "Ollama ready with model: %s", _OLLAMA_MODEL
            )
        else:
            logger.warning(
                "Ollama running but %s not pulled yet. "
                "Run: ollama pull %s",
                _OLLAMA_MODEL, _OLLAMA_MODEL
            )
        return found
    except Exception:
        return False


def init_groq():
    """
    Called once from load_models() at Django startup.
    Auto-selects best available tier.
    """
    global _groq_client, _groq_available
    global _ollama_available, _active_tier

    # ── Tier 1: Groq ─────────────────────────────────────
    if _check_internet():
        try:
            from groq import Groq
            from django.conf import settings
            api_key = (
                getattr(settings, 'GROQ_API_KEY', '')
                or os.environ.get('GROQ_API_KEY', '')
            )
            if api_key:
                _groq_client    = Groq(api_key=api_key)
                _groq_available = True
                _active_tier    = "groq"
                logger.info(
                    "AI Tier 1: Groq online (%s)",
                    _GROQ_MODEL
                )
                return
            else:
                logger.warning("GROQ_API_KEY not set")
        except Exception as exc:
            logger.warning("Groq init failed: %s", exc)
    else:
        logger.info("No internet — skipping Groq")

    # ── Tier 2: Ollama ───────────────────────────────────
    if _check_ollama():
        _ollama_available = True
        _active_tier      = "ollama"
        logger.info(
            "AI Tier 2: Ollama local offline (%s)",
            _OLLAMA_MODEL
        )
        return

    # ── Tier 3: Rule-based ───────────────────────────────
    _active_tier = "rules"
    logger.info(
        "AI Tier 3: Rule-based fallback active "
        "(no Groq, no Ollama)"
    )


def get_active_tier():
    return _active_tier or "rules"


def is_available():
    return True   # always available via rules fallback


# ════════════════════════════════════════════════════════
# CALL HELPERS
# ════════════════════════════════════════════════════════

def _call_groq(system: str, user: str,
               retries: int = 2) -> Optional[str]:
    if not _groq_available or not _groq_client:
        return None
    for attempt in range(1, retries + 1):
        try:
            resp = _groq_client.chat.completions.create(
                model    = _GROQ_MODEL,
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature     = 0.1,
                max_tokens      = 512,
                response_format = {"type": "json_object"},
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            err = str(exc)
            if "429" in err or "rate" in err.lower():
                wait = 10 * attempt
                logger.warning(
                    "Groq rate limited, waiting %ds", wait
                )
                time.sleep(wait)
            else:
                logger.warning(
                    "Groq call error (attempt %d): %s",
                    attempt, exc
                )
                break
    return None


def _call_ollama(system: str, user: str) -> Optional[str]:
    if not _ollama_available:
        return None
    try:
        import urllib.request
        payload = json.dumps({
            "model":   _OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream":  False,
            "format":  "json",
            "options": {"temperature": 0.1},
        }).encode()

        req = urllib.request.Request(
            _OLLAMA_URL,
            data    = payload,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
            return (
                data.get("message", {})
                    .get("content", "")
            )
    except Exception as exc:
        logger.warning("Ollama call error: %s", exc)
        return None


def _call_ai(system: str, user: str) -> Optional[str]:
    """
    Route to best available tier.
    Auto-falls back if primary tier fails mid-session.
    """
    tier = _active_tier or "rules"

    if tier == "groq":
        result = _call_groq(system, user)
        if result:
            return result
        # Lost internet mid-session — try Ollama
        logger.info(
            "Groq failed mid-session, trying Ollama fallback"
        )
        if _check_ollama():
            return _call_ollama(system, user)
        return None

    elif tier == "ollama":
        result = _call_ollama(system, user)
        if result:
            return result
        # Ollama stopped — try Groq if internet appeared
        if _check_internet() and _groq_client:
            logger.info(
                "Ollama failed, trying Groq fallback"
            )
            return _call_groq(system, user)
        return None

    return None   # rules tier uses no AI calls


def _safe_json(text: Optional[str]) -> Optional[Dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


# ════════════════════════════════════════════════════════
# TIER 3: RULE-BASED FALLBACK
# ════════════════════════════════════════════════════════

_OCR_FIXES = [
    (re.compile(r'\b([A-Za-z]+)0([A-Za-z]+)\b'), r'\g<1>o\g<2>'),
    (re.compile(r'\brn\b'), 'm'),
    (re.compile(r'\bcl\b'), 'd'),
]

_NOT_MEDICINES = {
    'the','and','for','take','with','after','before','food',
    'water','days','weeks','times','daily','morning','night',
    'evening','doctor','patient','name','age','date','uhid',
    'hospital','signature','adequate','fluid','intake','bp',
    'pr','rbs','temp','stat','giddiness','feebleness',
    'hypoglycemic','nausea','fever','cough','cold','pain',
    'ache','vomiting','diarrhoea','diarrhea','weakness',
}

def _rules_correct_ocr(text: str) -> str:
    for pattern, replacement in _OCR_FIXES:
        text = pattern.sub(replacement, text)
    return text


def _rules_extract_medicines(ocr_text: str) -> List[Dict]:
    """
    Extract medicine names using patterns only — no AI.
    Looks for lines that contain known medicine patterns.
    """
    medicines = []
    seen = set()

    lines = [
        l.strip() for l in ocr_text.split('\n')
        if l.strip() and len(l.strip()) > 2
    ]

    # Patterns that suggest a medicine line
    medicine_patterns = [
        # "Tab. Paracetamol 500mg"
        re.compile(
            r'\b(tab\.?|cap\.?|inj\.?|syr\.?|'
            r'inf\.?|susp\.?|oint\.?)\s+([A-Za-z][\w\s-]+)',
            re.IGNORECASE
        ),
        # Lines with dosage like "500mg", "5ml", "5%"
        re.compile(
            r'\b([A-Za-z][A-Za-z\s-]{2,}?)\s+'
            r'(\d+(?:\.\d+)?)\s*(%|mg|ml|g|mcg|iu)\b',
            re.IGNORECASE
        ),
        # "ORS 2 sachets"
        re.compile(
            r'\b([A-Za-z]{3,})\s+\d+\s+sachets?\b',
            re.IGNORECASE
        ),
    ]

    # Route patterns
    route_patterns = {
        'IV':   re.compile(r'\b(iv|intravenous)\b', re.I),
        'IM':   re.compile(r'\b(im|intramuscular)\b', re.I),
        'oral': re.compile(r'\b(oral|po|mouth)\b', re.I),
    }

    for line in lines:
        for pattern in medicine_patterns:
            m = pattern.search(line)
            if m:
                # Get the medicine name from first group
                name = m.group(1).strip() if \
                       m.lastindex >= 1 else ""

                # Skip prefixes (Tab., Cap., etc.)
                if re.match(
                    r'^(tab|cap|inj|syr|inf|susp|oint)\.?$',
                    name, re.I
                ):
                    if m.lastindex >= 2:
                        name = m.group(2).strip()

                name = re.sub(r'\s+', ' ', name).strip()

                # Skip if too short or is a stopword
                if (len(name) < 3 or
                        name.lower() in _NOT_MEDICINES):
                    continue

                key = name.lower()
                if key in seen:
                    continue
                seen.add(key)

                # Detect route
                route = None
                for r_name, r_pat in route_patterns.items():
                    if r_pat.search(line):
                        route = r_name
                        break

                medicines.append({
                    "name":       name,
                    "route":      route,
                    "dosage":     None,
                    "is_medicine": True,
                })
                break   # one match per line

    return medicines


def _rules_validate_match(
    ocr_text: str,
    matched_name: str,
) -> Dict[str, Any]:
    """
    Simple rule-based match validation.
    Checks if key words from matched name appear in OCR.
    """
    ocr_lower = ocr_text.lower()

    # Extract significant words from matched name
    # (skip generic words like "tablet", "mg", "syrup")
    skip = {
        'tablet','tablets','capsule','capsules','syrup',
        'injection','infusion','solution','suspension',
        'cream','gel','ointment','drops','sachet',
        'mg','ml','mcg','g','iu','ip','bp','usp',
    }
    name_words = [
        w.lower() for w in re.findall(r'[a-zA-Z]+', matched_name)
        if w.lower() not in skip and len(w) >= 3
    ]

    if not name_words:
        return {
            "match_valid":       True,
            "match_explanation": "Match accepted (rule-based).",
            "confidence_label":  "Medium",
        }

    # Count how many significant words appear in OCR
    hits = sum(1 for w in name_words if w in ocr_lower)
    ratio = hits / len(name_words)

    if ratio >= 0.5:
        confidence = "High" if ratio >= 0.8 else "Medium"
        return {
            "match_valid":       True,
            "match_explanation": "Match accepted (rule-based).",
            "confidence_label":  confidence,
        }
    else:
        return {
            "match_valid":       False,
            "match_explanation": (
                f"OCR text may not match '{matched_name}'. "
                f"Please verify manually."
            ),
            "confidence_label":  "Low",
        }


# ════════════════════════════════════════════════════════
# VALIDATION FUNCTIONS
# ════════════════════════════════════════════════════════

def validate_ocr_and_match(
    ocr_text:     str,
    matched_name: str,
    matched_salt: str,
) -> Dict[str, Any]:
    """
    Validate OCR output and database match.
    Routes to AI or rules depending on active tier.
    """
    default = {
        "ocr_corrected":     ocr_text,
        "match_valid":       True,
        "match_explanation": "Match accepted.",
        "confidence_label":  "Medium",
        "corrected_name":    None,
    }

    tier = _active_tier or "rules"

    # ── AI path (Groq or Ollama) ──────────────────────────
    if tier in ("groq", "ollama"):
        system = (
            "You are a clinical pharmacist AI validating "
            "medicine OCR results from Indian medicine "
            "strips and prescriptions. "
            "Respond with valid JSON only. No extra text."
        )
        user = f"""Medicine scan OCR validation:

RAW OCR TEXT:
\"\"\"{ocr_text}\"\"\"

DATABASE MATCH:
  Medicine : {matched_name}
  Salt     : {matched_salt}

Tasks:
1. Fix obvious OCR errors (O/0, l/1, rn/m, missing spaces)
2. Is the database match plausible for the OCR text?
3. If clearly wrong, suggest the correct medicine name.

Reply ONLY with this JSON:
{{
  "ocr_corrected":     "<cleaned OCR text>",
  "match_valid":       <true or false>,
  "match_explanation": "<one sentence>",
  "confidence_label":  "<High, Medium, or Low>",
  "corrected_name":    "<better name or null>"
}}"""

        raw    = _call_ai(system, user)
        parsed = _safe_json(raw)

        if parsed:
            return {
                "ocr_corrected":
                    str(parsed.get("ocr_corrected", ocr_text)),
                "match_valid":
                    bool(parsed.get("match_valid", True)),
                "match_explanation":
                    str(parsed.get("match_explanation", "")),
                "confidence_label":
                    str(parsed.get("confidence_label", "Medium")),
                "corrected_name":
                    parsed.get("corrected_name"),
            }
        # AI call failed — fall through to rules

    # ── Rules path ────────────────────────────────────────
    corrected = _rules_correct_ocr(ocr_text)
    validation = _rules_validate_match(corrected, matched_name)
    return {
        "ocr_corrected":     corrected,
        "match_valid":       validation["match_valid"],
        "match_explanation": validation["match_explanation"],
        "confidence_label":  validation["confidence_label"],
        "corrected_name":    None,
    }


def validate_alternatives(
    detected_name:  str,
    detected_salt:  str,
    detected_price: Optional[float],
    alternatives:   List[Dict],
) -> List[Dict]:
    """
    Filter alternatives list using AI or rules.
    """
    if not alternatives:
        return []

    tier = _active_tier or "rules"

    # ── AI path ───────────────────────────────────────────
    if tier in ("groq", "ollama"):
        candidates = [
            {
                "index": i,
                "name":  a.get("name", ""),
                "salt":  a.get("salt", ""),
                "price": a.get("price", 0),
            }
            for i, a in enumerate(alternatives[:8])
        ]

        system = (
            "You are a clinical pharmacist AI filtering "
            "generic medicine alternatives for bioequivalence "
            "in India. Respond with valid JSON only."
        )
        user = f"""Detected medicine:
  Name  : {detected_name}
  Salt  : {detected_salt}
  Price : Rs.{detected_price or 'unknown'}

Candidate alternatives:
{json.dumps(candidates, indent=2)}

Keep only alternatives with the SAME active ingredient.
Remove irrelevant entries. Sort cheapest first. Max 5.

Reply ONLY with this JSON:
{{
  "valid_indices":  [<index numbers, cheapest first>],
  "removed_reason": "<reason if any removed, else null>"
}}"""

        raw    = _call_ai(system, user)
        parsed = _safe_json(raw)

        if parsed and "valid_indices" in parsed:
            removed = parsed.get("removed_reason")
            if removed:
                logger.info(
                    "AI removed alternatives: %s", removed
                )
            filtered = [
                alternatives[i]
                for i in parsed.get("valid_indices", [])
                if isinstance(i, int)
                and 0 <= i < len(alternatives)
            ]
            if filtered:
                return filtered[:5]
        # AI failed — fall through to rules

    # ── Rules path: return as-is (already sorted cheapest) ─
    return alternatives[:5]


def extract_medicines_with_groq(
    ocr_text:    str,
    groq_client: Any,
    model:       str,
) -> tuple:
    """
    Extract medicine names from prescription OCR text.
    Uses AI if available, falls back to rules.
    Returns (medicines_list, diagnosis_string).
    """
    tier = _active_tier or "rules"

    # ── AI path ───────────────────────────────────────────
    if tier in ("groq", "ollama"):
        system = (
            "You are a clinical pharmacist AI reading Indian "
            "doctor prescriptions. Extract ONLY medicine names. "
            "Ignore diagnosis, vitals, patient info, hospital "
            "name, general advice. Respond with valid JSON only."
        )
        user = f"""Prescription OCR text:
\"\"\"{ocr_text}\"\"\"

Extract ONLY medicines/drugs prescribed.
Ignore: patient name/age/UHID, hospital name, date,
diagnosis (c/o, imp), vitals (BP, PR, RBS, temp),
general advice (fluid intake, rest, diet),
doctor signature, regional language text.

Reply ONLY with this JSON:
{{
  "medicines": [
    {{
      "name":        "<medicine name>",
      "route":       "<oral|IV|IM|topical|null>",
      "dosage":      "<dosage or null>",
      "is_medicine": <true or false>
    }}
  ],
  "raw_diagnosis": "<patient complaints, else null>"
}}"""

        raw    = _call_ai(system, user)
        parsed = _safe_json(raw)

        if parsed and "medicines" in parsed:
            meds = [
                m for m in parsed["medicines"]
                if m.get("is_medicine", True)
                and len(m.get("name", "").strip()) >= 3
            ]
            diag = parsed.get("raw_diagnosis") or ""
            logger.info(
                "AI extracted %d medicines from prescription",
                len(meds)
            )
            return meds, diag
        # AI failed — fall through to rules

    # ── Rules path ────────────────────────────────────────
    logger.info(
        "Using rule-based medicine extraction (tier: %s)", tier
    )
    meds = _rules_extract_medicines(ocr_text)
    return meds, ""


# ════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════

def run_ai_validation(
    scan_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main function called from scan_image().
    Never raises — always returns result dict.
    """
    tier = get_active_tier()

    if not scan_result.get("success"):
        scan_result["ai_validation"] = {
            "available": True,
            "tier":      tier,
        }
        return scan_result

    try:
        medicines    = scan_result.get("medicines", [])
        alternatives = scan_result.get("alternatives", [])
        ocr_text     = scan_result.get("ocr_text", "")

        if not medicines:
            scan_result["ai_validation"] = {
                "available": True,
                "tier":      tier,
                "skipped":   True,
            }
            return scan_result

        top = medicines[0]

        # Validate OCR + match
        ocr_result = validate_ocr_and_match(
            ocr_text     = ocr_text,
            matched_name = top.get("name", ""),
            matched_salt = top.get("salt", ""),
        )

        # Filter alternatives
        if alternatives:
            scan_result["alternatives"] = validate_alternatives(
                detected_name  = top.get("name", ""),
                detected_salt  = top.get("salt", ""),
                detected_price = top.get("price"),
                alternatives   = alternatives,
            )

        scan_result["ai_validation"] = {
            "available":         True,
            "tier":              tier,
            "confidence_label":  ocr_result["confidence_label"],
            "match_valid":       ocr_result["match_valid"],
            "match_explanation": ocr_result["match_explanation"],
            "ocr_corrected":     ocr_result["ocr_corrected"],
            "corrected_name":    ocr_result["corrected_name"],
        }

        if (not ocr_result["match_valid"]
                and ocr_result["corrected_name"]):
            scan_result["ai_suggested_correction"] = \
                ocr_result["corrected_name"]

        logger.info(
            "AI validation done — tier=%s valid=%s conf=%s",
            tier,
            ocr_result["match_valid"],
            ocr_result["confidence_label"],
        )

    except Exception as exc:
        logger.error(
            "run_ai_validation error: %s", exc, exc_info=True
        )
        scan_result["ai_validation"] = {
            "available": True,
            "tier":      tier,
            "error":     str(exc),
        }

    return scan_result
