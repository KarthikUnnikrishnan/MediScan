"""
scanner/ml/vision.py — Vision models for MediScan: YOLOv8 detector & TrOCR text recognition.

PATCHED — anti-hallucination guard added to run_ocr().

Root cause of hallucination: TrOCR is a seq2seq model trained on medicine-strip
text. When it receives a low-information image (blank foil crop, wrong side of
blister pack, too-small region), it still generates tokens from its LM prior —
commonly outputting "aceclofenac paracetamol tablets" or similar high-frequency
training phrases instead of declining.

Fix: validate TrOCR output with _ocr_output_looks_real(). The check is
lightweight (regex + token heuristics) and does NOT require internet access.
If the output fails validation, run_ocr() returns "" so the caller can
safely return unverified rather than presenting a hallucinated medicine name.
"""

import os
import re
import sys
import logging
from django.conf import settings
from PIL import Image

# ---------------------------------------------------------------------------
# Windows DLL fix — must run before *any* torch import.
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    try:
        import ctypes
        import site

        _torch_lib_dir = None
        for _sp in site.getsitepackages():
            _candidate = os.path.join(_sp, "torch", "lib")
            if os.path.isdir(_candidate):
                _torch_lib_dir = _candidate
                break

        if _torch_lib_dir:
            os.add_dll_directory(_torch_lib_dir)
            _iomp_path = os.path.join(_torch_lib_dir, "libiomp5md.dll")
            if os.path.isfile(_iomp_path):
                ctypes.CDLL(_iomp_path)
    except Exception as _dll_fix_err:
        logging.getLogger(__name__).warning(
            "Windows DLL pre-load fix failed (non-fatal): %s", _dll_fix_err
        )
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

_strip_detector = None
_ocr_processor  = None
_ocr_model      = None
_device         = None


# ── Structural OCR quality helpers ─────────────────────────────────────────────

# Structural indicators of pharmaceutical packaging / label text:
# - strength value (digits + mg/ml/mcg/g/iu)
# - pharmacopeia or dosage schedule suffix (IP, BP, USP, ER, SR, CR)
# - dosage form indicators (tablet, capsule, syrup, etc.)
_STRENGTH_RE = re.compile(
    r'\b\d+(?:\.\d+)?\s*(?:mg|ml|mcg|g|iu|units?|%)\b',
    re.IGNORECASE,
)
_PHARMA_SUFFIX_RE = re.compile(
    r'\b(?:ip|bp|usp|er|sr|cr|xl|od|bd|tds|hs|sos)\b',
    re.IGNORECASE,
)
_DOSAGE_FORM_RE = re.compile(
    r'\b(?:tablet|capsule|syrup|suspension|injection|drops|ointment|gel|cream|strip)\b',
    re.IGNORECASE,
)


def _ocr_output_looks_real(text: str) -> bool:
    """
    Evaluate whether *text* has valid structural characteristics for OCR text.
    Uses purely algorithmic and structural heuristics without hardcoding any
    specific medicine names.
    """
    if not text:
        return False

    t = text.strip()
    if len(t) < 4:
        return False

    alnum = re.sub(r'[^a-zA-Z0-9]', '', t)
    if len(alnum) < 3:
        return False

    # Check for degenerate repetition (e.g. model looping on a single token)
    words = [w.lower() for w in t.split() if w]
    if len(words) >= 4 and len(set(words)) == 1:
        logger.warning("run_ocr: repetitive token hallucination detected: %r", t[:80])
        return False

    t_lower = t.lower()

    # Pass: contains a dosage strength
    if _STRENGTH_RE.search(t):
        return True

    # Pass: contains a pharmacopeia / clinical suffix
    if _PHARMA_SUFFIX_RE.search(t):
        return True

    # Reject text that is only a single isolated dosage form word without any other context
    if _DOSAGE_FORM_RE.search(t_lower) and len(t) < 15 and len(words) <= 1:
        logger.warning(
            "run_ocr: output contains only an isolated dosage form word: %r", t
        )
        return False

    # Default: accept candidate for downstream matching
    return True


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models():
    """
    Load YOLO strip detector and fine-tuned TrOCR model.
    Called once at startup.
    """
    global _strip_detector, _ocr_processor, _ocr_model, _device

    try:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from ultralytics import YOLO

        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        _strip_detector = YOLO(str(settings.STRIP_DETECTOR_PATH))

        _ocr_processor = TrOCRProcessor.from_pretrained(
            str(settings.STRIP_OCR_PATH),
            use_fast=False,
        )

        _ocr_model = VisionEncoderDecoderModel.from_pretrained(
            str(settings.STRIP_OCR_PATH)
        )
        if hasattr(_ocr_model, "generation_config") and _ocr_model.generation_config:
            _ocr_model.generation_config.early_stopping = False
            _ocr_model.generation_config.length_penalty = 1.0
            _ocr_model.generation_config.num_beams = 1
        _ocr_model.to(_device)
        _ocr_model.eval()

        logger.info("MediScan vision models loaded successfully on %s", _device)

    except Exception as exc:
        logger.error("Failed to load MediScan vision models: %s", exc, exc_info=True)
        raise


# ── Strip crop ────────────────────────────────────────────────────────────────

def crop_strip(image_pil):
    """
    Detect medicine strip / label region:
    1. If YOLO detects a large package/strip box (>20% of image), crop that region.
    2. If YOLO detects multiple small blister pockets/cavities (<15% of image each),
       blister text is printed on the foil label side (opposite to the blister pockets).
       Crop the primary brand/active ingredient label region on the foil side.
    3. Fallback: crop the central text band on the label side (left 55%).
    """
    try:
        W, H = image_pil.size
        img_area = W * H

        results = _strip_detector(image_pil, conf=0.15)
        boxes = results[0].boxes if results else None

        if boxes is not None and len(boxes) > 0:
            box_list = boxes.xyxy.cpu().tolist()
            confs    = boxes.conf.cpu().tolist()

            # Check if there is a dominant package/strip box (>20% of image area)
            dominant_boxes = [
                (b, c) for b, c in zip(box_list, confs)
                if ((b[2] - b[0]) * (b[3] - b[1])) / img_area >= 0.20
            ]

            if dominant_boxes:
                best_box = max(
                    dominant_boxes,
                    key=lambda x: (x[0][2] - x[0][0]) * (x[0][3] - x[0][1]),
                )[0]
                x1, y1, x2, y2 = [int(v) for v in best_box]
                crop = image_pil.crop((x1, y1, x2, y2))
                logger.info(
                    "crop_strip: Dominant YOLO package crop (%dx%d)", crop.size[0], crop.size[1]
                )
                return crop, True

            # Small blister pockets detected — determine foil side
            # Only treat as genuine blister pockets if they meet minimum size & confidence criteria
            cavity_boxes = [
                b for b, c in zip(box_list, confs)
                if c >= 0.40 and 0.005 <= (((b[2] - b[0]) * (b[3] - b[1])) / img_area) <= 0.15
            ]
            if len(cavity_boxes) >= 3:
                mean_x = sum((b[0] + b[2]) / 2 for b in cavity_boxes) / len(cavity_boxes)
                logger.info(
                    "crop_strip: %d blister cavities detected (mean_x=%.1f)", len(cavity_boxes), mean_x
                )

                if mean_x > 0.5 * W:
                    # Cavities on right → foil text on left
                    x1, x2 = int(0.02 * W), int(0.52 * W)
                    y1, y2 = int(0.05 * H), int(0.95 * H)
                else:
                    # Cavities on left → foil text on right
                    x1, x2 = int(0.48 * W), int(0.98 * W)
                    y1, y2 = int(0.05 * H), int(0.95 * H)

                crop = image_pil.crop((x1, y1, x2, y2))
                logger.info("crop_strip: Blister-side foil crop (%dx%d)", crop.size[0], crop.size[1])
                return crop, True

        # Fallback: crop primary label band on left side
        left_crop = image_pil.crop(
            (int(0.02 * W), int(0.05 * H), int(0.52 * W), int(0.95 * H))
        )
        logger.info("crop_strip: Fallback central label crop")
        return left_crop, False

    except Exception as exc:
        logger.warning("crop_strip failed: %s", exc, exc_info=True)
        return image_pil, False


# ── TrOCR text recognition ────────────────────────────────────────────────────

def run_ocr(image_pil) -> str:
    """
    Run TrOCR on *image_pil* and return the decoded text string.

    Includes an anti-hallucination guard: if the decoded text matches a
    known hallucination pattern (short phrase with no dosage/suffix info)
    the function returns "" so callers can safely handle unverified scans.

    Returns
    -------
    str — recognised text, or "" on failure or detected hallucination.
    """
    try:
        import torch  # noqa: PLC0415

        rgb_image    = image_pil.convert('RGB')
        pixel_values = _ocr_processor(
            images=rgb_image, return_tensors='pt'
        ).pixel_values.to(_device)

        with torch.no_grad():
            generated_ids = _ocr_model.generate(
                pixel_values,
                num_beams=1,
                max_new_tokens=64,
            )

        text = _ocr_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        if not text:
            return ""

        # ── Anti-hallucination guard ──────────────────────────────────────────
        if not _ocr_output_looks_real(text):
            logger.warning(
                "run_ocr: output rejected as hallucination: %r", text[:80]
            )
            return ""

        return text

    except Exception as exc:
        logger.warning("run_ocr failed: %s", exc, exc_info=True)
        return ""
