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

        results = _strip_detector(image_pil, conf=0.25)
        boxes = results[0].boxes if results else None

        if boxes is not None and len(boxes) > 0:
            box_list = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist()

            # Check if there is a dominant package/strip box (>20% of image area)
            dominant_boxes = [
                (b, c) for b, c in zip(box_list, confs)
                if ((b[2] - b[0]) * (b[3] - b[1])) / img_area >= 0.20
            ]

            if dominant_boxes:
                best_box = max(dominant_boxes, key=lambda x: (x[0][2] - x[0][0]) * (x[0][3] - x[0][1]))[0]
                x1, y1, x2, y2 = [int(v) for v in best_box]
                crop = image_pil.crop((x1, y1, x2, y2))
                logger.info("crop_strip: Dominant YOLO package crop used (%dx%d)", crop.size[0], crop.size[1])
                return crop, True

            # If small blister pockets were detected (each < 15% of image area)
            mean_x = sum((b[0] + b[2]) / 2 for b in box_list) / len(box_list)
            logger.info("crop_strip: Detected %d blister cavities (mean_x=%.1f)", len(box_list), mean_x)

            if mean_x > 0.5 * W:
                # Cavities on the right -> foil text is on the left
                # Primary medicine name / brand band is in center-lower region of foil
                x1, x2 = int(0.07 * W), int(0.52 * W)
                y1, y2 = int(0.45 * H), int(0.82 * H)
                crop = image_pil.crop((x1, y1, x2, y2))
                logger.info("crop_strip: Cavities on right -> foil text crop used (%dx%d)", crop.size[0], crop.size[1])
                return crop, True
            elif mean_x < 0.5 * W:
                # Cavities on the left -> foil text is on the right
                x1, x2 = int(0.48 * W), int(0.93 * W)
                y1, y2 = int(0.45 * H), int(0.82 * H)
                crop = image_pil.crop((x1, y1, x2, y2))
                logger.info("crop_strip: Cavities on left -> foil text crop used (%dx%d)", crop.size[0], crop.size[1])
                return crop, True

        # Fallback: crop primary label band on left side
        left_crop = image_pil.crop((int(0.07 * W), int(0.45 * H), int(0.52 * W), int(0.82 * H)))
        logger.info("crop_strip: Fallback central label crop used")
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
