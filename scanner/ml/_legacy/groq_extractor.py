"""
scanner/ml/groq_extractor.py — Groq cloud NLP extraction for MediScan.

Sends prescription / OCR text to the Groq API and extracts a clean list
of medicine names using an instruction-tuned LLaMA model.

Model cascade (primary → fallback):
  1. llama-3.3-70b-versatile  — current recommended model
  2. llama3-70b-8192           — stable fallback
  3. llama-3.1-8b-instant     — lightweight last-resort

FIX: llama-3.1-70b-versatile was decommissioned by Groq on 2026-09-xx.
     The replacement is llama-3.3-70b-versatile (same capability tier,
     new weight version). This file always tries the current cascade so
     a future decommission auto-degrades gracefully instead of crashing.
"""

import os
import json
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Model cascade ─────────────────────────────────────────────────────────────
_GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # Current recommended — replaces decommissioned 3.1-70b
    "llama3-70b-8192",            # Stable fallback
    "llama-3.1-8b-instant",       # Lightweight last-resort
]

# ── Prompt ────────────────────────────────────────────────────────────────────
_SYSTEM_MSG = (
    "You are a pharmaceutical assistant. "
    "Your only job is to extract medicine names from text. "
    "Return a JSON array of strings — each string is exactly one medicine name "
    "as it appears in the text. "
    "Do NOT include dosage, frequency, or instructions. "
    "Do NOT guess medicines that are not in the text. "
    'If no medicines are found return an empty array: []'
)


def extract_medicines_with_groq(
    ocr_text: str,
    api_key: Optional[str] = None,
    max_medicines: int = 10,
) -> List[str]:
    """
    Extract medicine names from *ocr_text* using the Groq API.

    Parameters
    ----------
    ocr_text : str
        Raw OCR / prescription text.
    api_key : str, optional
        Groq API key (falls back to GROQ_API_KEY env var).
    max_medicines : int
        Maximum number of medicine names to return.

    Returns
    -------
    list[str]
        Extracted medicine names, or [] on any failure.
    """
    if not ocr_text or not ocr_text.strip():
        return []

    if api_key is None:
        api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        logger.warning("extract_medicines_with_groq: GROQ_API_KEY not set — skipping")
        return []

    try:
        from groq import Groq  # noqa: PLC0415
    except ImportError:
        logger.warning("extract_medicines_with_groq: groq package not installed")
        return []

    user_msg = f"Extract medicine names from this text:\n\n{ocr_text[:2000]}"

    last_error = None
    for model_name in _GROQ_MODELS:
        client = None
        try:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": _SYSTEM_MSG},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            raw = response.choices[0].message.content or ""

            # Parse the JSON array from the response
            medicines = _parse_json_list(raw)
            if medicines is not None:
                logger.info(
                    "extract_medicines_with_groq: %s extracted %d medicines",
                    model_name,
                    len(medicines),
                )
                return medicines[:max_medicines]

            logger.warning(
                "extract_medicines_with_groq: %s returned unparseable output: %r",
                model_name,
                raw[:200],
            )
            return []

        except Exception as exc:
            last_error = exc
            err_str = str(exc).lower()

            # Decommissioned / unsupported model — try next
            if (
                "decommissioned" in err_str
                or "deprecated" in err_str
                or "not found" in err_str
                or "does not exist" in err_str
                or "400" in err_str
            ):
                logger.warning(
                    "extract_medicines_with_groq: %s unavailable/decommissioned, "
                    "trying next model",
                    model_name,
                )
                continue

            # Rate limit / server overload — try next model
            if "429" in err_str or "503" in err_str or "rate" in err_str:
                logger.warning(
                    "extract_medicines_with_groq: %s → rate-limit/overload, trying next",
                    model_name,
                )
                continue

            # Any other error — log and bail
            logger.error(
                "extract_medicines_with_groq: %s → unexpected error: %s",
                model_name,
                exc,
            )
            return []

        finally:
            # Explicit client cleanup to avoid resource-leak warnings
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    if last_error:
        logger.error(
            "extract_medicines_with_groq: all models exhausted. Last: %s", last_error
        )
    return []


def _parse_json_list(text: str) -> Optional[List[str]]:
    """
    Extract a JSON array from *text*, which may contain prose around it.

    Returns list[str] on success, None on parse failure.
    """
    import re  # noqa: PLC0415

    text = text.strip()

    # Try whole text first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except json.JSONDecodeError:
        pass

    # Try to extract the first JSON array embedded in prose
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except json.JSONDecodeError:
            pass

    return None
