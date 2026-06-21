"""
Model 2 — Prescription HTR Pipeline
-------------------------------------
Step 1: EasyOCR reads all text regions from the full prescription page
Step 2: Low-confidence regions are re-read by fine-tuned TrOCR
Step 3: Medicine candidate lines identified by dose/prefix patterns
Step 4: rapidfuzz corrects each candidate against 352K medicine database
"""

import os, re, cv2, torch, numpy as np, sqlite3
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
from rapidfuzz import process, fuzz

# ── PATHS ─────────────────────────────────────────────────────────────
HTR_MODEL_DIR = r"D:\Coding Section\Mediscan\saved_models\prescription_htr"
MEDICINE_DB   = r"D:\Coding Section\Mediscan\db_cache\medicines.sqlite"

# ── GLOBALS ───────────────────────────────────────────────────────────
_trocr_processor = None
_trocr_model     = None
_ocr_reader      = None
_medicine_names  = []
_device          = None


def load_models():
    global _trocr_processor, _trocr_model, _ocr_reader, _medicine_names, _device

    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[PrescriptionHTR] Loading on: {_device}")

    # Fine-tuned TrOCR (from Colab training)
    if os.path.exists(HTR_MODEL_DIR):
        _trocr_processor = TrOCRProcessor.from_pretrained(HTR_MODEL_DIR)
        _trocr_model     = VisionEncoderDecoderModel.from_pretrained(
                               HTR_MODEL_DIR
                           ).to(_device)
        _trocr_model.eval()
        print("[PrescriptionHTR] TrOCR fine-tuned model loaded")
    else:
        print("[PrescriptionHTR] Warning: TrOCR model folder not found")
        print(f"                  Expected: {HTR_MODEL_DIR}")
        print("                  Continuing with EasyOCR only")

    # EasyOCR for full-page text detection
    _ocr_reader = easyocr.Reader(['en'], gpu=(_device == 'cuda'), verbose=False)
    print("[PrescriptionHTR] EasyOCR loaded")

    # Load medicine names for fuzzy correction
    if os.path.exists(MEDICINE_DB):
        conn  = sqlite3.connect(MEDICINE_DB)
        rows  = conn.execute(
            "SELECT DISTINCT name FROM medicines WHERE name IS NOT NULL"
        ).fetchall()
        conn.close()
        _medicine_names = [r[0].strip() for r in rows if r[0] and len(r[0]) > 2]
        print(f"[PrescriptionHTR] {len(_medicine_names):,} medicine names loaded for correction")
    else:
        print("[PrescriptionHTR] Warning: medicines.sqlite not found — correction disabled")

    print("[PrescriptionHTR] Ready.\n")


# ── HELPERS ───────────────────────────────────────────────────────────

def _preprocess(image_bgr):
    """Enhance contrast on prescription image for better OCR."""
    gray     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def _trocr_read(crop_bgr):
    """Run fine-tuned TrOCR on a single word/line crop."""
    if _trocr_model is None:
        return ""
    try:
        pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        pv  = _trocr_processor(pil, return_tensors='pt').pixel_values.to(_device)
        with torch.no_grad():
            ids = _trocr_model.generate(pv, max_new_tokens=24)
        return _trocr_processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    except Exception:
        return ""


def _fuzzy_correct(word, threshold=72):
    """
    Match OCR word against 352K medicine names.
    Even imperfect OCR ('Rotril') gets corrected to 'Rivotril'.
    """
    if not _medicine_names or len(word) < 3:
        return word, 0
    result = process.extractOne(word, _medicine_names, scorer=fuzz.WRatio)
    if result and result[1] >= threshold:
        return result[0], result[1]
    return word, 0


# Patterns to identify medicine lines in a prescription
_DOSE_RE   = re.compile(r'\b\d+\s*(?:mg|mcg|ml|g|iu|%)\b', re.I)
_PREFIX_RE = re.compile(
    r'\b(tab|cap|caps|inj|syr|syp|susp|oint|gel|drop|cream|sol)\.?\b', re.I
)


def _extract_medicine_candidates(text_list):
    """
    From a list of OCR text strings, identify medicine name candidates.
    A line is a medicine if it has a dose pattern or Tab/Cap/Inj prefix.
    Standalone CamelCase or ALLCAPS words are possible brand names.
    """
    candidates = []
    for text in text_list:
        text = text.strip()
        if len(text) < 2:
            continue

        if _DOSE_RE.search(text) or _PREFIX_RE.search(text):
            # Strip prefix and dose to isolate the drug name
            name = _PREFIX_RE.sub('', text)
            name = _DOSE_RE.sub('', name).strip(' .,-:')
            if name and len(name) > 2:
                candidates.append(name)

        elif re.match(r'^[A-Z][a-z]{3,}$|^[A-Z]{4,15}$', text.strip()):
            # Standalone brand name (e.g. "Rivotril", "AUGMENTIN")
            candidates.append(text.strip())

    return candidates


# ── PUBLIC API ────────────────────────────────────────────────────────

def predict(image_path: str) -> dict:
    """
    Main entry point called from Django views.

    Pipeline:
        1. Load & preprocess prescription image
        2. EasyOCR detects all text regions
        3. TrOCR re-reads low-confidence regions
        4. Extract medicine candidate lines
        5. Fuzzy-correct each against 352K medicine DB
        6. Return corrected medicine list

    Returns:
        {
            "success"         : True / False,
            "medicines_found" : ["Levocetirizine 5mg", "Augmentin 625"],
            "all_text"        : ["raw", "ocr", "lines", ...],
            "error"           : ""
        }
    """
    if _ocr_reader is None:
        return {
            "success"        : False,
            "medicines_found": [],
            "all_text"       : [],
            "error"          : "Models not loaded. Call load_models() first.",
        }

    # ── 1. Read image ──────────────────────────────────────────────────
    image = cv2.imread(image_path)
    if image is None:
        return {
            "success"        : False,
            "medicines_found": [],
            "all_text"       : [],
            "error"          : "Could not read image file.",
        }

    # ── 2. Preprocess ──────────────────────────────────────────────────
    enhanced = _preprocess(image)

    # ── 3. EasyOCR full-page pass ──────────────────────────────────────
    raw = _ocr_reader.readtext(enhanced, detail=1, paragraph=False)

    all_texts     = []
    refined_texts = []

    for (bbox, text, conf) in raw:
        text = text.strip()
        if len(text) < 2:
            continue
        all_texts.append(text)

        if conf < 0.55 and _trocr_model is not None:
            # Low confidence — try TrOCR on this specific crop
            pts = np.array(bbox, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            pad  = 4
            x1   = max(0, x - pad)
            y1   = max(0, y - pad)
            x2   = min(image.shape[1], x + w + pad)
            y2   = min(image.shape[0], y + h + pad)
            crop = image[y1:y2, x1:x2]

            if crop.size > 0:
                trocr_text = _trocr_read(crop)
                # Use TrOCR result if it produced something meaningful
                refined_texts.append(
                    trocr_text if trocr_text and len(trocr_text) >= len(text) - 1
                    else text
                )
            else:
                refined_texts.append(text)
        else:
            # EasyOCR was confident — trust it
            refined_texts.append(text)

    # ── 4. Extract medicine candidates ────────────────────────────────
    candidates = _extract_medicine_candidates(refined_texts)

    if not candidates:
        return {
            "success"        : False,
            "medicines_found": [],
            "all_text"       : all_texts,
            "error"          : "No medicine names identified in the prescription.",
        }

    # ── 5. Fuzzy correct each candidate against medicine DB ───────────
    medicines_found = []
    for name in candidates:
        corrected, score = _fuzzy_correct(name)
        if corrected not in medicines_found:
            medicines_found.append(corrected)

    return {
        "success"        : True,
        "medicines_found": medicines_found,
        "all_text"       : all_texts,
        "error"          : "",
    }