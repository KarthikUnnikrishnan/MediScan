"""
scanner/ml/gemini_vision.py — Modern Gemini Vision OCR for MediScan.

Standardized on the modern `google-genai` SDK (`from google import genai`).
Submits medicine strip and package images to Gemini Vision to transcribe
visibly present label text (brand name, ingredients, strength, dosage form, manufacturer).

Configuration:
- GEMINI_API_KEY: loaded from Django settings or environment variable.
- GEMINI_MODEL: loaded from Django settings or environment variable (default: 'gemini-2.5-flash').
  Does NOT silently cascade through alternative models on request failure.
"""

import atexit
import io
import logging
import os
import re
import time
from typing import Any, Dict, Optional, Union
from PIL import Image

logger = logging.getLogger(__name__)

# ── Sentinel returned when Gemini genuinely cannot read text from image ─────
NO_TEXT_DETECTED = "NO_TEXT_DETECTED"

# ── Cached Client Instance ───────────────────────────────────────────────────
_client_instance: Optional[Any] = None
_client_api_key: Optional[str] = None


def get_gemini_model() -> str:
    """Return the configured Gemini model name."""
    env_model = os.environ.get('GEMINI_MODEL')
    if env_model:
        return env_model
    try:
        from django.conf import settings
        return getattr(settings, 'GEMINI_MODEL', '') or 'gemini-3.5-flash'
    except Exception:
        return 'gemini-3.5-flash'


def get_gemini_client(api_key: Optional[str] = None) -> Any:
    """
    Get or create a cached Google GenAI Client instance.
    Reuses the client across scans to avoid connection overhead.
    """
    global _client_instance, _client_api_key

    if not api_key:
        try:
            from django.conf import settings
            api_key = getattr(settings, 'GEMINI_API_KEY', '') or os.environ.get('GEMINI_API_KEY', '')
        except Exception:
            api_key = os.environ.get('GEMINI_API_KEY', '')

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    if _client_instance is None or _client_api_key != api_key:
        from google import genai
        _client_instance = genai.Client(api_key=api_key)
        _client_api_key = api_key

    return _client_instance


def _close_gemini_client() -> None:
    """Cleanly close the underlying client on process termination."""
    global _client_instance
    if _client_instance is not None:
        try:
            if hasattr(_client_instance, 'close'):
                _client_instance.close()
        except Exception:
            pass
        _client_instance = None


atexit.register(_close_gemini_client)


# ── Conservative Pharmaceutical Transcription Prompt ─────────────────────────
_SYSTEM_PROMPT = """\
You are an expert pharmaceutical label OCR assistant.
Carefully transcribe ONLY the text visibly printed on this medicine package, strip, or blister label.

Rules:
1. Transcribe ONLY what is visibly present in the image.
2. Do NOT guess, assume, or invent unreadable text, strengths, active ingredients, or manufacturers.
3. If any text is partially unreadable or blurred, transcribe only the legible letters and mark uncertain portions with [?].
4. Do NOT hallucinate a complete medicine identity from memory or partial text.
5. If absolutely NO text is visible or readable, output exactly: NO_TEXT_DETECTED

Format your output using these structured fields (one per line):
BRAND: <brand or trade name visible on label, or blank if not found>
GENERIC: <generic or active chemical name printed on label, or blank if not found>
INGREDIENTS: <active ingredient names with visible strengths and units, e.g. Levocetirizine Dihydrochloride (5mg), or blank>
STRENGTH: <dosage strength visible on label, e.g. 5mg, 500mg, or blank>
DOSAGE_FORM: <dosage form, e.g. Tablet, Capsule, Syrup, or blank>
MANUFACTURER: <manufacturer name visible on label, or blank>
SCHEDULE: <prescription schedule warning if visible, e.g. Schedule H, or blank>
RAW_TEXT: <any other visible text lines from the label>
"""


def _encode_image(image_input: Union[Image.Image, bytes, str]) -> bytes:
    """Convert PIL Image, file path, or bytes to JPEG bytes."""
    if isinstance(image_input, bytes):
        return image_input
    if isinstance(image_input, str):
        with open(image_input, "rb") as f:
            return f.read()
    if isinstance(image_input, Image.Image):
        buf = io.BytesIO()
        image_input.convert("RGB").save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def _extract_response_text(response: Any) -> str:
    """
    Extract text content from Gemini response.
    Silently filters out thought/reasoning tokens from thinking models
    so reasoning parts are never treated as OCR text.
    """
    try:
        parts = response.candidates[0].content.parts
        text_parts = []
        for part in parts:
            if hasattr(part, "thought") and part.thought:
                continue
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)
        return "".join(text_parts).strip()
    except (AttributeError, IndexError):
        pass

    try:
        return (response.text or "").strip()
    except Exception:
        return ""


def gemini_vision_ocr_structured(
    image_input: Union[Image.Image, bytes, str],
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send medicine image to Gemini Vision OCR using the modern google-genai SDK.

    Returns
    -------
    dict:
        {
            "success": bool,
            "raw_text": str,
            "plain_text": str,
            "structured": {
                "brand": str or None,
                "generic": str or None,
                "ingredients": str or None,
                "strength": str or None,
                "dosage_form": str or None,
                "manufacturer": str or None,
                "schedule": str or None,
            },
            "error": str or None,
            "error_type": str or None,
            "is_empty": bool,
            "model": str,
        }
    """
    model_name = get_gemini_model()
    empty_structured = {
        "brand": None,
        "generic": None,
        "ingredients": None,
        "strength": None,
        "dosage_form": None,
        "manufacturer": None,
        "schedule": None,
    }

    try:
        client = get_gemini_client(api_key=api_key)
    except RuntimeError as cfg_err:
        logger.warning("gemini_vision_ocr: configuration error: %s", cfg_err)
        return {
            "success": False,
            "raw_text": "",
            "plain_text": "",
            "structured": empty_structured,
            "error": str(cfg_err),
            "error_type": "CONFIG_ERROR",
            "is_empty": True,
            "model": model_name,
        }
    except Exception as exc:
        logger.error("gemini_vision_ocr: failed to initialize client: %s", type(exc).__name__)
        return {
            "success": False,
            "raw_text": "",
            "plain_text": "",
            "structured": empty_structured,
            "error": str(exc),
            "error_type": "INIT_ERROR",
            "is_empty": True,
            "model": model_name,
        }

    try:
        image_bytes = _encode_image(image_input)
    except Exception as enc_err:
        logger.warning("gemini_vision_ocr: failed to encode image: %s", enc_err)
        return {
            "success": False,
            "raw_text": "",
            "plain_text": "",
            "structured": empty_structured,
            "error": f"Image encoding error: {enc_err}",
            "error_type": "ENCODING_ERROR",
            "is_empty": True,
            "model": model_name,
        }

    try:
        from google.genai import types
        models = [model_name]
        try:
            from django.conf import settings
            fallback_value = getattr(settings, "GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite")
        except Exception:
            fallback_value = os.environ.get("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite")
        if isinstance(fallback_value, str):
            fallbacks = [item.strip() for item in fallback_value.split(",") if item.strip()]
        else:
            fallbacks = list(fallback_value or [])
        models.extend(model for model in fallbacks if model and model not in models)
        response = None
        last_exc = None
        for candidate_model in models:
            model_name = candidate_model
            retries = 0
            while True:
                try:
                    logger.info("gemini_vision_ocr: requesting OCR from model %s (%d bytes)", model_name, len(image_bytes))
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), _SYSTEM_PROMPT],
                    )
                    logger.info("gemini_vision_ocr: model %s worked", model_name)
                    break
                except Exception as exc:
                    last_exc = exc
                    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                    message = str(exc)
                    if status is None:
                        match = re.search(r"\b(400|401|403|404|429|5\d\d)\b", message)
                        status = int(match.group(1)) if match else None
                    logger.warning("gemini_vision_ocr: model %s failed (HTTP %s): %s", model_name, status, exc)
                    if status == 404 or "NOT_FOUND" in message:
                        break
                    if status in (429, 500, 501, 502, 503, 504) and retries < 2:
                        time.sleep(0.25 * (2 ** retries))
                        retries += 1
                        continue
                    break
            if response is not None:
                break
        if response is None:
            raise last_exc

        raw_text = _extract_response_text(response)

        if not raw_text or raw_text == NO_TEXT_DETECTED:
            logger.info("gemini_vision_ocr: model %s returned no readable text", model_name)
            return {
                "success": True,
                "raw_text": NO_TEXT_DETECTED,
                "plain_text": "",
                "structured": empty_structured,
                "error": None,
                "error_type": "NO_TEXT",
                "is_empty": True,
                "model": model_name,
            }

        # Parse structured fields
        structured = dict(empty_structured)
        field_patterns = {
            "brand": r'^BRAND:\s*(.*)$',
            "generic": r'^GENERIC:\s*(.*)$',
            "ingredients": r'^INGREDIENTS?:\s*(.*)$',
            "strength": r'^STRENGTH:\s*(.*)$',
            "dosage_form": r'^DOSAGE_FORM:\s*(.*)$',
            "manufacturer": r'^MANUFACTURER:\s*(.*)$',
            "schedule": r'^SCHEDULE:\s*(.*)$',
        }

        for line in raw_text.splitlines():
            line_str = line.strip()
            for field_key, pat in field_patterns.items():
                m = re.match(pat, line_str, flags=re.IGNORECASE)
                if m:
                    val = m.group(1).strip()
                    if val and val.lower() not in ("none", "null", "blank", "n/a"):
                        structured[field_key] = val

        plain_text = strip_preamble(raw_text)

        logger.info(
            "gemini_vision_ocr: success with %s (brand=%r, ingredients=%r)",
            model_name,
            structured["brand"],
            structured["ingredients"],
        )

        return {
            "success": True,
            "raw_text": raw_text,
            "plain_text": plain_text,
            "structured": structured,
            "error": None,
            "error_type": None,
            "is_empty": False,
            "model": model_name,
        }

    except Exception as exc:
        err_msg = str(exc)
        err_cls = type(exc).__name__
        logger.exception(
            "gemini_vision_ocr: model %s request failed (HTTP %s): %s (%s)",
            model_name,
            getattr(exc, "status_code", None) or getattr(exc, "code", None),
            err_cls,
            err_msg[:120],
        )
        return {
            "success": False,
            "raw_text": "",
            "plain_text": "",
            "structured": empty_structured,
            "error": f"{err_cls}: {err_msg}",
            "error_type": "API_ERROR",
            "is_empty": True,
            "model": model_name,
        }


def gemini_vision_ocr(
    image_input: Union[Image.Image, bytes, str],
    api_key: Optional[str] = None,
) -> str:
    """
    Main backward-compatible OCR entry point.
    Returns raw structured string from Gemini, or NO_TEXT_DETECTED, or "" on error.
    """
    res = gemini_vision_ocr_structured(image_input, api_key=api_key)
    if not res.get("success"):
        return ""
    return res.get("raw_text", "")


def strip_preamble(raw_gemini_text: str) -> str:
    """
    Remove field labels (BRAND:, INGREDIENTS:, etc.) from Gemini's response
    so the downstream OCR parser and matcher see clean text lines.
    """
    if not raw_gemini_text or raw_gemini_text == NO_TEXT_DETECTED:
        return ""

    lines = raw_gemini_text.strip().splitlines()
    cleaned = []
    for line in lines:
        line = re.sub(
            r'^\s*(?:BRAND|GENERIC|INGREDIENT(?:_\d+)?|INGREDIENTS|STRENGTH|DOSAGE_FORM|MANUFACTURER|SCHEDULE|RAW_TEXT)\s*:\s*',
            '',
            line,
            flags=re.IGNORECASE,
        ).strip()
        if line and line.upper() not in ("NONE", "NULL", "BLANK", "N/A"):
            cleaned.append(line)

    return "\n".join(cleaned)
