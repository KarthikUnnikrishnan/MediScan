"""
scanner/ml/recognisers.py — Two independent recognisers behind one unified interface.

- PrintedRecogniser (Model A):
    Wraps PrintedTrOCR from scanner/ml/printed_trocr.py.
    Loads fine-tuned Model A (fp16 on GPU).
    EasyOCR detects text boxes, TrOCR decodes each box line-by-line.
    Returns UPPERCASE text lines. Loaded once at server startup.

- HandwrittenRecogniser (Model B):
    Stub for doctor prescription HTR model.
    Checks saved_models/modelB_handwritten_ocr.
    If missing, returns {"ok": False, "message": "Handwritten model is not installed yet"}.
    Loaded only on-demand when handwritten mode is requested; unloadable to free GPU VRAM.
"""

import os
import gc
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from PIL import Image

import torch
from django.conf import settings

logger = logging.getLogger(__name__)

_PRINTED_RECOGNISER_LOCK = threading.Lock()
_PRINTED_RECOGNISER: Optional["PrintedRecogniser"] = None

_HANDWRITTEN_RECOGNISER_LOCK = threading.Lock()
_HANDWRITTEN_RECOGNISER: Optional["HandwrittenRecogniser"] = None


class PrintedRecogniser:
    """
    Model A: Offline recognition engine for printed medicine blister strips and boxes.
    Uses EasyOCR for bounding box detection and fine-tuned TrOCR (fp16 on GPU) for recognition.
    """
    def __init__(self, model_dir: Optional[Union[str, Path]] = None, device: Optional[str] = None):
        self.model_dir = Path(model_dir or getattr(settings, 'MODEL_A_DIR', 'saved_models/modelA_printed_ocr'))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._inner: Optional[Any] = None
        self._init_lock = threading.Lock()

    def _ensure_loaded(self):
        if self._inner is None:
            with self._init_lock:
                if self._inner is None:
                    from .printed_trocr import PrintedTrOCR
                    logger.info("Initializing PrintedRecogniser (Model A) from %s on %s...", self.model_dir, self.device)
                    self._inner = PrintedTrOCR(model_dir=self.model_dir, device=self.device)
                    logger.info("PrintedRecogniser (Model A) ready.")

    def read_lines(self, photo: Union[str, Path, Image.Image], beams: int = 1, return_details: bool = False) -> Union[List[str], Tuple[List[str], Dict[str, Any]]]:
        """
        Reads all text lines from a photo.
        Returns list of UPPERCASE text strings (or tuple of lines and timing info if return_details is True).
        """
        self._ensure_loaded()
        return self._inner.read_lines(photo, beams=beams, return_details=return_details)

    def is_loaded(self) -> bool:
        return self._inner is not None


class HandwrittenRecogniser:
    """
    Model B: Offline recognition engine for handwritten doctor prescriptions.
    Operates on-demand and can be unloaded to free GPU VRAM.
    """
    def __init__(self, model_dir: Optional[Union[str, Path]] = None, device: Optional[str] = None):
        self.model_dir = Path(model_dir or getattr(settings, 'MODEL_B_DIR', 'saved_models/modelB_handwritten_ocr'))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._inner: Optional[Any] = None
        self._init_lock = threading.Lock()

    def is_model_installed(self) -> bool:
        """Check if Model B directory exists."""
        return self.model_dir.exists() and self.model_dir.is_dir()

    def _ensure_loaded(self):
        if self._inner is None:
            with self._init_lock:
                if self._inner is None:
                    from .handwritten_trocr import HandwrittenTrOCR
                    logger.info("Initializing HandwrittenRecogniser (Model B) from %s on %s...", self.model_dir, self.device)
                    self._inner = HandwrittenTrOCR(model_dir=self.model_dir, device=self.device)
                    logger.info("HandwrittenRecogniser (Model B) ready.")

    def read_prescription(
        self,
        photo: Union[str, Path, Image.Image],
        matcher: Optional[Any] = None,
        beams: int = 4,
        min_score: float = 80.0
    ) -> Dict[str, Any]:
        """
        Read handwritten prescription lines.
        Returns dict with keys ok, lines, medicines, message.
        """
        if not self.is_model_installed():
            return {
                "ok": False,
                "message": "Handwritten model is not installed yet",
                "lines": [],
                "medicines": []
            }

        self._ensure_loaded()
        return self._inner.read_prescription(photo, matcher=matcher, beams=beams, min_score=min_score)

    def unload(self):
        """Unload Model B from GPU memory."""
        with self._init_lock:
            if self._inner is not None:
                if hasattr(self._inner, "_det") and self._inner._det is not None:
                    del self._inner._det
                    self._inner._det = None
                del self._inner
                self._inner = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                logger.info("HandwrittenRecogniser (Model B) unloaded from GPU memory.")

    def is_loaded(self) -> bool:
        return self._inner is not None


def get_printed_recogniser() -> PrintedRecogniser:
    """Singleton getter for Model A."""
    global _PRINTED_RECOGNISER
    if _PRINTED_RECOGNISER is None:
        with _PRINTED_RECOGNISER_LOCK:
            if _PRINTED_RECOGNISER is None:
                _PRINTED_RECOGNISER = PrintedRecogniser()
    return _PRINTED_RECOGNISER


def get_handwritten_recogniser() -> HandwrittenRecogniser:
    """Singleton getter for Model B."""
    global _HANDWRITTEN_RECOGNISER
    if _HANDWRITTEN_RECOGNISER is None:
        with _HANDWRITTEN_RECOGNISER_LOCK:
            if _HANDWRITTEN_RECOGNISER is None:
                _HANDWRITTEN_RECOGNISER = HandwrittenRecogniser()
    return _HANDWRITTEN_RECOGNISER


def unload_handwritten_recogniser():
    """Explicitly free Model B resources."""
    global _HANDWRITTEN_RECOGNISER
    if _HANDWRITTEN_RECOGNISER is not None:
        with _HANDWRITTEN_RECOGNISER_LOCK:
            if _HANDWRITTEN_RECOGNISER is not None:
                _HANDWRITTEN_RECOGNISER.unload()
                _HANDWRITTEN_RECOGNISER = None


def warmup_printed_recogniser():
    """Pre-warm Model A in background thread at server startup."""
    try:
        recogniser = get_printed_recogniser()
        recogniser._ensure_loaded()
        logger.info("PrintedRecogniser warm-up complete.")
    except Exception as exc:
        logger.error("Failed to warm up PrintedRecogniser: %s", exc, exc_info=True)
