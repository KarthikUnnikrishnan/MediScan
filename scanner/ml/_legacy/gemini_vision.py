"""
scanner/ml/gemini_vision.py — Modern Gemini Vision OCR for MediScan.

Standardized on the modern `google-genai` SDK (`from google import genai`).
Submits medicine strip and package images to Gemini Vision to transcribe
visibly present label text (brand name, ingredients, strength, dosage form, manufacturer).

Key features:
- Central configuration: primary model (`gemini-3.1-flash-lite`), fallback models (`gemini-3.5-flash-lite`, `gemini-3.8-flash`).
- Intelligent HTTP 429 handling: distinguishes short-term rate limits (RPM/TPM) from daily quota exhaustion.
- Quota cooldown persistence: daily exhaustion status is saved to disk so restarting Django does not reset it.
- Intelligent HTTP 503 handling: bounded retries with exponential backoff (e.g. 2s, 4s), respecting server delays.
- Request timeout: configurable timeout prevents hanging on degraded connections.
- Duplicate request suppression: caches successful and transient failure results by image hash.
- Zero hallucination: transcription results are evidence for local database verification.
"""

import atexit
import hashlib
import io
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image

logger = logging.getLogger(__name__)

# ── Sentinel returned when Gemini genuinely cannot read text from image ─────
NO_TEXT_DETECTED = "NO_TEXT_DETECTED"

# ── Cached Client Instance ───────────────────────────────────────────────────
_client_instance: Optional[Any] = None
_client_api_key: Optional[str] = None
_client_lock = threading.Lock()

# ── In-Memory OCR & Negative Caches ──────────────────────────────────────────
_IMAGE_OCR_CACHE: Dict[str, Dict[str, Any]] = {}
_IMAGE_FAILURE_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_cache_lock = threading.Lock()

# ── Model Cooldown / Quota Tracking ──────────────────────────────────────────
_MODEL_COOLDOWNS: Dict[str, Dict[str, Any]] = {}
_quota_lock = threading.Lock()


def _get_quota_state_path() -> Path:
    """Return the filesystem path for persisting model quota/cooldown status."""
    try:
        from django.conf import settings
        path = getattr(settings, 'GEMINI_QUOTA_STATE_PATH', None)
        if path:
            return Path(path)
        base_dir = getattr(settings, 'BASE_DIR', Path('.'))
        return Path(base_dir) / 'saved_models' / 'gemini_quota_status.json'
    except Exception:
        return Path('saved_models/gemini_quota_status.json')


def _load_quota_state() -> Dict[str, Any]:
    """Load model cooldowns from disk."""
    state_path = _get_quota_state_path()
    if not state_path.exists():
        return {}
    try:
        with open(state_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('models', {})
    except Exception as exc:
        logger.warning("gemini_vision: could not load quota state from %s: %s", state_path, exc)
        return {}


def _save_quota_state(models_data: Dict[str, Any]) -> None:
    """Atomically save model cooldowns to disk."""
    state_path = _get_quota_state_path()
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = state_path.with_suffix('.tmp')
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump({'models': models_data}, f, indent=2)
        tmp_path.replace(state_path)
    except Exception as exc:
        logger.warning("gemini_vision: could not save quota state to %s: %s", state_path, exc)


def is_model_in_cooldown(model_name: str) -> Tuple[bool, Optional[str], float]:
    """
    Check if a model is currently in quota or temporary failure cooldown.
    Returns (in_cooldown, reason, cooldown_until_timestamp).
    """
    with _quota_lock:
        now = time.time()
        # Check memory first
        info = _MODEL_COOLDOWNS.get(model_name)
        if not info:
            # Fall back to disk state
            disk_state = _load_quota_state()
            _MODEL_COOLDOWNS.update(disk_state)
            info = _MODEL_COOLDOWNS.get(model_name)

        if info:
            cooldown_until = float(info.get('cooldown_until', 0.0))
            if now < cooldown_until:
                return True, info.get('reason', 'cooldown'), cooldown_until
            # Expired — clean up
            _MODEL_COOLDOWNS.pop(model_name, None)
            disk_state = _load_quota_state()
            if model_name in disk_state:
                disk_state.pop(model_name, None)
                _save_quota_state(disk_state)

    return False, None, 0.0


def set_model_cooldown(
    model_name: str,
    duration_seconds: float,
    reason: str,
    last_error: str = "",
) -> None:
    """
    Set a model into cooldown for duration_seconds.
    Persists to disk so restarting Django does not reset the quota cooldown.
    """
    with _quota_lock:
        now = time.time()
        cooldown_until = now + max(1.0, float(duration_seconds))
        entry = {
            'status': 'cooldown',
            'reason': reason,
            'cooldown_until': cooldown_until,
            'last_error': str(last_error)[:200],
            'last_updated': time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
        }
        _MODEL_COOLDOWNS[model_name] = entry
        disk_state = _load_quota_state()
        disk_state[model_name] = entry
        _save_quota_state(disk_state)

    logger.warning(
        "gemini_vision: placed model %s on cooldown for %.1fs (reason: %s)",
        model_name,
        duration_seconds,
        reason,
    )


def reset_model_cooldowns(model_name: Optional[str] = None) -> None:
    """Reset cooldowns for a single model or all models."""
    with _quota_lock:
        if model_name:
            _MODEL_COOLDOWNS.pop(model_name, None)
            disk_state = _load_quota_state()
            if model_name in disk_state:
                disk_state.pop(model_name, None)
                _save_quota_state(disk_state)
            logger.info("gemini_vision: reset cooldown for model %s", model_name)
        else:
            _MODEL_COOLDOWNS.clear()
            _save_quota_state({})
            logger.info("gemini_vision: reset all model cooldowns")


def get_all_model_statuses() -> Dict[str, Any]:
    """Return current cooldown / quota status for all tracked models."""
    with _quota_lock:
        disk_state = _load_quota_state()
        _MODEL_COOLDOWNS.update(disk_state)
        now = time.time()
        result = {}
        for m, info in list(_MODEL_COOLDOWNS.items()):
            until = float(info.get('cooldown_until', 0.0))
            if now < until:
                result[m] = {
                    'in_cooldown': True,
                    'remaining_seconds': round(until - now, 1),
                    'reason': info.get('reason'),
                    'last_error': info.get('last_error'),
                }
            else:
                result[m] = {'in_cooldown': False}
        return result


def clear_image_ocr_cache() -> None:
    """Clear the in-memory image OCR cache."""
    with _cache_lock:
        _IMAGE_OCR_CACHE.clear()
        _IMAGE_FAILURE_CACHE.clear()


# ── Configuration Getters ───────────────────────────────────────────────────

def get_gemini_model() -> str:
    """Return the configured primary Gemini model name."""
    env_model = os.environ.get('GEMINI_MODEL')
    try:
        from django.conf import settings
        settings_model = getattr(settings, 'GEMINI_MODEL', None)
        if settings_model:
            if env_model and env_model != 'gemini-3.1-flash-lite' and str(settings_model).strip() == 'gemini-3.1-flash-lite':
                return env_model.strip()
            return str(settings_model).strip()
    except Exception:
        pass

    if env_model:
        return env_model.strip()

    return 'gemini-3.1-flash-lite'


def get_gemini_fallback_models() -> List[str]:
    """Return the configured list of fallback Gemini models in priority order."""
    try:
        from django.conf import settings
        settings_val = getattr(settings, 'GEMINI_FALLBACK_MODELS', None)
        if settings_val is not None:
            if isinstance(settings_val, str):
                return [m.strip() for m in settings_val.split(',') if m.strip()]
            return list(settings_val)
    except Exception:
        pass

    env_fallbacks = os.environ.get('GEMINI_FALLBACK_MODELS')
    if env_fallbacks is not None:
        return [m.strip() for m in env_fallbacks.split(',') if m.strip()]

    return ['gemini-3.5-flash-lite', 'gemini-3.8-flash']


def get_gemini_timeout() -> float:
    """Return the per-request timeout in seconds."""
    env_timeout = os.environ.get('GEMINI_REQUEST_TIMEOUT')
    if env_timeout:
        try:
            return float(env_timeout)
        except ValueError:
            pass
    try:
        from django.conf import settings
        return float(getattr(settings, 'GEMINI_REQUEST_TIMEOUT', 10.0))
    except Exception:
        return 10.0


def get_max_503_retries() -> int:
    """Return the maximum retries for HTTP 503 service overload."""
    env_retries = os.environ.get('GEMINI_MAX_503_RETRIES')
    if env_retries:
        try:
            return int(env_retries)
        except ValueError:
            pass
    try:
        from django.conf import settings
        return int(getattr(settings, 'GEMINI_MAX_503_RETRIES', 2))
    except Exception:
        return 2


def get_503_backoff_base() -> float:
    """Return the base delay in seconds for exponential backoff on HTTP 503."""
    env_base = os.environ.get('GEMINI_503_BACKOFF_BASE')
    if env_base:
        try:
            return float(env_base)
        except ValueError:
            pass
    try:
        from django.conf import settings
        return float(getattr(settings, 'GEMINI_503_BACKOFF_BASE', 2.0))
    except Exception:
        return 2.0


def get_rate_limit_retries() -> int:
    """Return the maximum retries for short-term RPM rate limiting."""
    env_retries = os.environ.get('GEMINI_RATE_LIMIT_RETRIES')
    if env_retries:
        try:
            return int(env_retries)
        except ValueError:
            pass
    try:
        from django.conf import settings
        return int(getattr(settings, 'GEMINI_RATE_LIMIT_RETRIES', 1))
    except Exception:
        return 1


def get_daily_cooldown_hours() -> float:
    """Return default cooldown duration in hours for daily quota exhaustion."""
    env_cd = os.environ.get('GEMINI_DAILY_COOLDOWN_HOURS')
    if env_cd:
        try:
            return float(env_cd)
        except ValueError:
            pass
    try:
        from django.conf import settings
        return float(getattr(settings, 'GEMINI_DAILY_COOLDOWN_HOURS', 24.0))
    except Exception:
        return 24.0


# ── Client Lifecycle ────────────────────────────────────────────────────────

def _is_client_closed(client: Any) -> bool:
    """Check if the underlying HTTP client has been closed."""
    if client is None:
        return True
    try:
        api_client = getattr(client, '_api_client', None)
        http_client = getattr(api_client, '_httpx_client', None)
        if http_client is not None and getattr(http_client, 'is_closed', False):
            return True
    except Exception:
        pass
    return False


def get_gemini_client(api_key: Optional[str] = None) -> Any:
    """
    Get or create a cached Google GenAI Client instance.
    Reuses the client across scans, automatically recreating if closed.
    """
    global _client_instance, _client_api_key

    with _client_lock:
        if not api_key:
            try:
                from django.conf import settings
                api_key = getattr(settings, 'GEMINI_API_KEY', '') or os.environ.get('GEMINI_API_KEY', '')
            except Exception:
                api_key = os.environ.get('GEMINI_API_KEY', '')

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        if _client_instance is None or _client_api_key != api_key or _is_client_closed(_client_instance):
            from google import genai
            _client_instance = genai.Client(api_key=api_key)
            _client_api_key = api_key

        return _client_instance


def _close_gemini_client() -> None:
    """Cleanly reset the client instance."""
    global _client_instance, _client_api_key
    with _client_lock:
        if _client_instance is not None:
            try:
                if hasattr(_client_instance, 'close'):
                    _client_instance.close()
            except Exception:
                pass
            _client_instance = None
            _client_api_key = None


# ── Conservative Pharmaceutical Transcription Prompt ─────────────────────────
_SYSTEM_PROMPT = """\
You are an expert pharmaceutical label OCR assistant.
Carefully transcribe ONLY the text visibly printed on this medicine package, strip, or blister label.

Critical distinction:
- PRODUCT TRADE NAME / BRAND: The specific commercial trade name of the medicine (e.g. "P-650", "1-AL", "Dolo 650", "Augmentin 625 Duo", "Calpol 500", "Crocin 650").
  * CRITICAL: Do NOT transcribe the pharmaceutical company name or corporate logo (e.g. "Apex", "Cipla", "Micro Labs", "Sun Pharma", "FDC", "Abbott") as the BRAND if there is a specific product trade name (e.g. "P-650" or "1-AL"). Put the company name in MANUFACTURER.
  * For alphanumeric codes and numbers (e.g. "P-650", "1-AL", "PCM-650"), transcribe letters and numbers exactly as printed (do NOT confuse digit '1' with 'T' or 'I').
- GENERIC: The pharmacopeia generic title (e.g. "Paracetamol Tablets IP", "Levocetirizine Tablets IP 5 mg").
- MANUFACTURER: The company that manufactured or marketed the product (e.g. "Apex Laboratories Pvt Ltd", "FDC Limited").

Rules:
1. Transcribe ONLY what is visibly present in the image.
2. Do NOT guess, assume, or invent unreadable text, strengths, active ingredients, or manufacturers.
3. If any text is partially unreadable or blurred, transcribe only the legible letters and mark uncertain portions with [?].
4. Do NOT hallucinate a complete medicine identity from memory or partial text.
5. If absolutely NO text is visible or readable, output exactly: NO_TEXT_DETECTED

Format your output using these structured fields (one per line):
BRAND: <exact commercial product trade name printed on label, e.g. P-650, 1-AL, Dolo 650, or blank if unbranded>
GENERIC: <generic or active chemical name printed on label, or blank if not found>
INGREDIENTS: <active ingredient names with visible strengths and units, e.g. Levocetirizine (5mg) or Paracetamol IP (650mg), or blank>
STRENGTH: <dosage strength visible on label, e.g. 650mg, 5mg, or blank>
DOSAGE_FORM: <dosage form, e.g. Tablet, Capsule, Syrup, or blank>
MANUFACTURER: <company or manufacturer name visible on label, e.g. Apex Laboratories, or blank>
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
    Filters out thought/reasoning tokens so reasoning parts are never treated as OCR text.
    """
    try:
        parts = response.candidates[0].content.parts
        text_parts = []
        for part in parts:
            if hasattr(part, "thought") and part.thought:
                continue
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)
        if text_parts:
            return "".join(text_parts).strip()
    except (AttributeError, IndexError):
        pass

    try:
        return (response.text or "").strip()
    except Exception:
        return ""


def _parse_error_details(exc: Exception) -> Dict[str, Any]:
    """
    Inspect exception to extract status code, error status, message, and retry delay.
    Classifies 429 daily quota exhaustion vs RPM rate limit vs 503 unavailable vs auth errors.
    """
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    status_text = getattr(exc, "status", None)
    message = str(getattr(exc, "message", None) or exc or "")
    details = getattr(exc, "details", None)

    if status_code is None:
        m = re.search(r"\b(400|401|403|404|429|500|502|503|504)\b", message)
        if m:
            status_code = int(m.group(1))

    # Extract retry delay if provided
    retry_delay: Optional[float] = None
    if isinstance(details, dict):
        err_obj = details.get("error", details)
        for item in err_obj.get("details", []):
            if isinstance(item, dict) and "retryDelay" in item:
                d_str = str(item["retryDelay"])
                m_del = re.match(r"^([0-9.]+)\s*s?$", d_str)
                if m_del:
                    try:
                        retry_delay = float(m_del.group(1))
                    except ValueError:
                        pass
    if retry_delay is None:
        m_del = re.search(r"retry(?:Delay|_delay)[\s\"':=]+([0-9.]+)\s*s?", message, re.IGNORECASE)
        if m_del:
            try:
                retry_delay = float(m_del.group(1))
            except ValueError:
                pass

    lower_msg = message.lower()
    details_str = json.dumps(details).lower() if isinstance(details, (dict, list)) else ""
    full_err_text = f"{lower_msg} {details_str}"

    is_daily_quota = bool(
        status_code == 429 and (
            "perday" in full_err_text or
            "generaterequestsperday" in full_err_text or
            "daily" in full_err_text or
            (retry_delay is not None and retry_delay >= 3600.0)
        )
    )

    is_rate_limit = bool(
        status_code == 429 and not is_daily_quota
    )

    is_unavailable = bool(
        status_code in (500, 502, 503, 504) or
        "unavailable" in full_err_text or
        status_text == "UNAVAILABLE"
    )

    is_auth_error = bool(
        status_code in (401, 403) or
        "permission_denied" in full_err_text or
        "api_key_invalid" in full_err_text or
        status_text == "PERMISSION_DENIED"
    )

    is_not_found = bool(
        status_code == 404 or
        "not_found" in full_err_text or
        status_text == "NOT_FOUND"
    )

    is_timeout = bool(
        "timeout" in lower_msg or
        "timed out" in lower_msg
    )

    return {
        "status_code": status_code,
        "status": status_text,
        "message": message,
        "retry_delay": retry_delay,
        "is_daily_quota": is_daily_quota,
        "is_rate_limit": is_rate_limit,
        "is_unavailable": is_unavailable,
        "is_auth_error": is_auth_error,
        "is_not_found": is_not_found,
        "is_timeout": is_timeout,
    }


def gemini_vision_ocr_structured(
    image_input: Union[Image.Image, bytes, str],
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send medicine image to Gemini Vision OCR with intelligent model fallback,
    bounded retries, separate per-model quota tracking, and response caching.

    Order:
    1. Primary model (default: gemini-3.1-flash-lite)
    2. Fallback models (default: gemini-3.5-flash-lite, gemini-3.8-flash)
    """
    primary_model = get_gemini_model()
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
            "model": primary_model,
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
            "model": primary_model,
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
            "model": primary_model,
        }

    # ── Check Duplicate Image Caches ──────────────────────────────────────────
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    with _cache_lock:
        if image_hash in _IMAGE_OCR_CACHE:
            cached = dict(_IMAGE_OCR_CACHE[image_hash])
            cached["cached"] = True
            logger.info("gemini_vision_ocr: returning cached OCR result for hash %s", image_hash[:10])
            return cached

        # Check negative failure cache (within last 30s)
        fail_entry = _IMAGE_FAILURE_CACHE.get(image_hash)
        if fail_entry:
            fail_time, fail_res = fail_entry
            if time.time() - fail_time < 30.0:
                logger.info("gemini_vision_ocr: reusing recent failure result for identical image %s", image_hash[:10])
                return fail_res

    try:
        from google.genai import types

        # Build prioritized list of model candidates
        models = [primary_model]
        fallbacks = get_gemini_fallback_models()
        for fb in fallbacks:
            if fb and fb not in models:
                models.append(fb)

        timeout_sec = get_gemini_timeout()
        max_503 = get_max_503_retries()
        backoff_base = get_503_backoff_base()
        max_rate_retries = get_rate_limit_retries()
        daily_cooldown_hours = get_daily_cooldown_hours()

        response = None
        successful_model = None
        last_error_details: Dict[str, Any] = {}
        last_error_model = primary_model

        for candidate_model in models:
            # 1. Skip models currently under quota or service cooldown
            in_cd, cd_reason, cd_until = is_model_in_cooldown(candidate_model)
            if in_cd:
                logger.info(
                    "gemini_vision_ocr: skipping model %s (on %s cooldown until %s)",
                    candidate_model,
                    cd_reason,
                    time.strftime("%H:%M:%S", time.localtime(cd_until)),
                )
                continue

            last_error_model = candidate_model
            retries_503 = 0
            retries_rate = 0
            client_closed_retried = False

            # Model generation configuration with strict timeout and single attempt (MediScan controls retries)
            model_config = None
            try:
                http_options = types.HttpOptions(
                    timeout=int(timeout_sec * 1000),
                    retry_options=types.HttpRetryOptions(attempts=1),
                )
                if "3.8" in candidate_model or "thinking" in candidate_model:
                    model_config = types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                        temperature=0.1,
                        http_options=http_options,
                    )
                else:
                    model_config = types.GenerateContentConfig(
                        temperature=0.1,
                        http_options=http_options,
                    )
            except Exception:
                pass

            while True:
                try:
                    if _is_client_closed(client):
                        client = get_gemini_client(api_key=api_key)

                    logger.info(
                        "gemini_vision_ocr: requesting OCR from model %s (%d bytes)",
                        candidate_model,
                        len(image_bytes),
                    )
                    kwargs = {
                        "model": candidate_model,
                        "contents": [
                            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                            _SYSTEM_PROMPT,
                        ],
                    }
                    if model_config is not None:
                        kwargs["config"] = model_config

                    response = client.models.generate_content(**kwargs)
                    successful_model = candidate_model
                    logger.info("gemini_vision_ocr: model %s succeeded", candidate_model)
                    break

                except Exception as exc:
                    parsed = _parse_error_details(exc)
                    last_error_details = parsed
                    logger.warning(
                        "gemini_vision_ocr: model %s failed (HTTP %s, %s): %s",
                        candidate_model,
                        parsed["status_code"],
                        parsed["status"],
                        parsed["message"][:120],
                    )

                    # Re-initialize if client closed
                    if "client has been closed" in parsed["message"].lower() and not client_closed_retried:
                        client_closed_retried = True
                        _close_gemini_client()
                        client = get_gemini_client(api_key=api_key)
                        continue

                    # Authentication errors (401, 403, PERMISSION_DENIED) — permanent, never retry
                    if parsed["is_auth_error"]:
                        logger.error("gemini_vision_ocr: permanent authentication error: %s", parsed["message"])
                        return {
                            "success": False,
                            "raw_text": "",
                            "plain_text": "",
                            "structured": empty_structured,
                            "error": parsed["message"],
                            "error_type": "AUTH_ERROR",
                            "is_empty": True,
                            "model": candidate_model,
                        }

                    # HTTP 429 Daily Quota Exhaustion — record persistent daily cooldown and move to next model
                    if parsed["is_daily_quota"]:
                        cd_duration = parsed["retry_delay"] if parsed["retry_delay"] and parsed["retry_delay"] > 0 else (daily_cooldown_hours * 3600.0)
                        set_model_cooldown(candidate_model, cd_duration, "daily_quota_exhausted", parsed["message"])
                        break

                    # HTTP 429 Short-term RPM/TPM Rate Limit — bounded retry if delay reasonable
                    if parsed["is_rate_limit"]:
                        if parsed["retry_delay"] is not None and parsed["retry_delay"] <= 15.0 and retries_rate < max_rate_retries:
                            sleep_dur = max(1.0, parsed["retry_delay"])
                            logger.info("gemini_vision_ocr: rate limit on %s — sleeping %.1fs (attempt %d/%d)", candidate_model, sleep_dur, retries_rate + 1, max_rate_retries)
                            time.sleep(sleep_dur)
                            retries_rate += 1
                            continue
                        # If delay too long or retries exhausted, set short cooldown and move to fallback
                        cd_dur = parsed["retry_delay"] if parsed["retry_delay"] and parsed["retry_delay"] > 0 else 60.0
                        set_model_cooldown(candidate_model, cd_dur, "rate_limit_exceeded", parsed["message"])
                        break

                    # HTTP 503 Service Overload / 5xx Unavailable — bounded exponential backoff
                    if parsed["is_unavailable"]:
                        if retries_503 < max_503:
                            delay = max(parsed["retry_delay"] or 0.0, backoff_base * (2 ** retries_503))
                            logger.info(
                                "gemini_vision_ocr: 503 unavailable for %s — retrying in %.1fs (attempt %d/%d)",
                                candidate_model,
                                delay,
                                retries_503 + 1,
                                max_503,
                            )
                            time.sleep(delay)
                            retries_503 += 1
                            continue
                        # Retries exhausted — set temporary cooldown (120s) and fall back
                        set_model_cooldown(candidate_model, 120.0, "service_unavailable", parsed["message"])
                        break

                    # HTTP 404 Model Not Found — mark unsupported and move to fallback
                    if parsed["is_not_found"]:
                        set_model_cooldown(candidate_model, 86400.0, "model_not_found", parsed["message"])
                        break

                    # Other error (e.g. timeout) — move to fallback
                    break

            if response is not None:
                break

        if response is None:
            err_type = "API_ERROR"
            if last_error_details.get("is_daily_quota"):
                err_type = "QUOTA_EXHAUSTED"
            elif last_error_details.get("is_unavailable"):
                err_type = "SERVICE_UNAVAILABLE"
            elif last_error_details.get("is_rate_limit"):
                err_type = "RATE_LIMITED"
            elif last_error_details.get("is_timeout"):
                err_type = "TIMEOUT"
            elif last_error_details.get("is_not_found"):
                err_type = "NOT_FOUND"

            fail_result = {
                "success": False,
                "raw_text": "",
                "plain_text": "",
                "structured": empty_structured,
                "error": last_error_details.get("message") or "All configured Gemini models failed or in cooldown.",
                "error_type": err_type,
                "is_empty": True,
                "model": last_error_model,
            }
            with _cache_lock:
                _IMAGE_FAILURE_CACHE[image_hash] = (time.time(), fail_result)
            return fail_result

        # Process received response
        raw_text = _extract_response_text(response)

        if not raw_text or raw_text == NO_TEXT_DETECTED:
            logger.info("gemini_vision_ocr: model %s returned no readable text", successful_model)
            res = {
                "success": True,
                "raw_text": NO_TEXT_DETECTED,
                "plain_text": "",
                "structured": empty_structured,
                "error": None,
                "error_type": "NO_TEXT",
                "is_empty": True,
                "model": successful_model,
            }
            with _cache_lock:
                _IMAGE_OCR_CACHE[image_hash] = res
            return res

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

        # Trade name disambiguation and recovery from raw lines
        brand_val = (structured.get("brand") or "").strip()
        mfr_val = (structured.get("manufacturer") or "").strip()
        gen_val = (structured.get("generic") or "").strip()

        is_mfr_brand = bool(
            brand_val and mfr_val and (
                brand_val.lower() in mfr_val.lower() or
                mfr_val.lower().startswith(brand_val.lower())
            )
        )
        is_generic_brand = bool(
            brand_val and (
                brand_val.lower() in ("paracetamol", "levocetirizine", "ibuprofen", "amoxicillin", "pantoprazole", "cetirizine", "azithromycin") or
                (gen_val and (brand_val.lower() in gen_val.lower() or gen_val.lower().startswith(brand_val.lower())))
            )
        )

        recovered_trade_name = None
        for line in raw_text.splitlines():
            clean_l = re.sub(r'^(?:BRAND|GENERIC|INGREDIENTS?|STRENGTH|DOSAGE_FORM|MANUFACTURER|SCHEDULE|RAW_TEXT):\s*', '', line.strip(), flags=re.I).strip()
            # 1. Alphanumeric trade codes like P-650, 1-AL, PCM-650, Pan-D, Meftal-P, D-650
            m_code = re.search(r'\b([A-Za-z0-9]{1,6}-[A-Za-z0-9]{1,6}(?:\s*(?:Tablets?|Capsules?|Suspension|Drops?))?)\b', clean_l, re.I)
            if m_code:
                cand = m_code.group(1).strip()
                if not re.match(r'^(?:mg|ml|mcg|kg|iu|ph|lic|no|lot|exp|mfg)$', cand, re.I):
                    recovered_trade_name = cand
                    break
            # 2. Brand + Strength like Dolo 650, Calpol 500, Crocin 650, Augmentin 625
            m_brand_num = re.search(r'\b([A-Za-z]{3,15}\s+\d{2,4}(?:\s*(?:Tablets?|Capsules?))?)\b', clean_l, re.I)
            if m_brand_num:
                cand = m_brand_num.group(1).strip()
                first_w = cand.split()[0].lower()
                generic_words = {'paracetamol', 'levocetirizine', 'ibuprofen', 'amoxicillin', 'azithromycin', 'pantoprazole', 'cetirizine', 'tablets', 'tablet', 'capsule'}
                if first_w not in generic_words and (not mfr_val or first_w not in mfr_val.lower()):
                    recovered_trade_name = cand
                    break

        if recovered_trade_name and (not brand_val or is_mfr_brand or is_generic_brand):
            logger.info("gemini_vision_ocr: recovered specific trade name %r (was brand=%r)", recovered_trade_name, brand_val)
            structured["brand"] = recovered_trade_name

        structured["detected_label_name"] = structured.get("brand") or recovered_trade_name or brand_val or gen_val
        plain_text = strip_preamble(raw_text)

        logger.info(
            "gemini_vision_ocr: success with %s (brand=%r, ingredients=%r)",
            successful_model,
            structured["brand"],
            structured["ingredients"],
        )

        res = {
            "success": True,
            "raw_text": raw_text,
            "plain_text": plain_text,
            "structured": structured,
            "error": None,
            "error_type": None,
            "is_empty": False,
            "model": successful_model,
        }
        with _cache_lock:
            _IMAGE_OCR_CACHE[image_hash] = res
        return res

    except Exception as exc:
        err_msg = str(exc)
        err_cls = type(exc).__name__
        logger.exception("gemini_vision_ocr: unexpected failure: %s", err_msg)
        return {
            "success": False,
            "raw_text": "",
            "plain_text": "",
            "structured": empty_structured,
            "error": f"{err_cls}: {err_msg}",
            "error_type": "API_ERROR",
            "is_empty": True,
            "model": primary_model,
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
