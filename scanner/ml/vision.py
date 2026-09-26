"""
scanner/ml/vision.py — Vision models for MediScan: YOLOv8 detector & TrOCR text recognition.

Keep YOLO and TrOCR models unchanged.
"""

import os
import sys
import logging
from django.conf import settings
from PIL import Image

# ---------------------------------------------------------------------------
# Windows DLL fix — must run before *any* torch import.
#
# Root cause: Oracle Database XE ships a stale libiomp5md.dll (build 20190109)
# into C:\app\Admin\product\21c\dbhomeXE\bin\ which sits on the system PATH.
# When torch's c10.dll loads, Windows resolves libiomp5md.dll from Oracle's
# directory first, causing the DLL initialization routine to fail (WinError 1114).
#
# Fix: register torch's own lib directory as a DLL search path (Python 3.8+
# os.add_dll_directory API), then explicitly ctypes-load torch's libiomp5md.dll
# before Windows has a chance to load the Oracle version from PATH.
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    try:
        import ctypes
        import site

        # Locate torch's lib directory inside the active venv / site-packages.
        _torch_lib_dir = None
        for _sp in site.getsitepackages():
            _candidate = os.path.join(_sp, "torch", "lib")
            if os.path.isdir(_candidate):
                _torch_lib_dir = _candidate
                break

        if _torch_lib_dir:
            # Register as a trusted DLL directory (takes precedence over PATH).
            os.add_dll_directory(_torch_lib_dir)

            # Pre-load torch's libiomp5md.dll so Windows won't later load
            # the Oracle-installed version from PATH.
            _iomp_path = os.path.join(_torch_lib_dir, "libiomp5md.dll")
            if os.path.isfile(_iomp_path):
                ctypes.CDLL(_iomp_path)
    except Exception as _dll_fix_err:
        # Non-fatal — log and continue; torch import may still succeed.
        logging.getLogger(__name__).warning(
            "Windows DLL pre-load fix failed (non-fatal): %s", _dll_fix_err
        )
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

_strip_detector = None
_ocr_processor = None
_ocr_model = None
_device = None


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
            str(settings.STRIP_OCR_PATH)
        )

        _ocr_model = VisionEncoderDecoderModel.from_pretrained(
            str(settings.STRIP_OCR_PATH)
        )
        _ocr_model.to(_device)
        _ocr_model.eval()

        logger.info("MediScan vision models loaded successfully on %s", _device)

    except Exception as exc:
        logger.error("Failed to load MediScan vision models: %s", exc, exc_info=True)
        raise


def crop_strip(image_pil):
    """
    Attempt YOLO detection first.
    If detection succeeds: use the detected crop.
    If detection fails: crop the left 60% of the image
    (blister strip text is typically on the label side).
    """
    try:
        results = _strip_detector(image_pil, conf=0.25)
        boxes = results[0].boxes if results else None

        if boxes is not None and len(boxes) > 0:
            confidences = boxes.conf.cpu().tolist()
            best_idx = confidences.index(max(confidences))
            x1, y1, x2, y2 = [
                int(v) for v in boxes.xyxy[best_idx].cpu().tolist()
            ]
            crop = image_pil.crop((x1, y1, x2, y2))
            logger.info("crop_strip: YOLO crop used")
            return crop, True

        W, H = image_pil.size
        left_crop = image_pil.crop((0, 0, int(W * 0.60), H))
        logger.info("crop_strip: YOLO miss — using left 60%% crop")
        return left_crop, False

    except Exception as exc:
        logger.warning("crop_strip failed: %s", exc, exc_info=True)
        return image_pil, False


def run_ocr(image_pil):
    """
    Run TrOCR on *image_pil* and return the decoded text string.
    """
    try:
        import torch

        rgb_image = image_pil.convert('RGB')
        pixel_values = _ocr_processor(
            images=rgb_image, return_tensors='pt'
        ).pixel_values.to(_device)

        with torch.no_grad():
            generated_ids = _ocr_model.generate(
                pixel_values,
                num_beams=4,
                max_new_tokens=128,
            )

        text = _ocr_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return text.strip()

    except Exception as exc:
        logger.warning("run_ocr failed: %s", exc, exc_info=True)
        return ""
