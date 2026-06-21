"""
Model 1 — Strip OCR Pipeline  (EasyOCR edition)
------------------------------------------------
YOLOv8 confirms a medicine strip is present (gating) →
EasyOCR reads ALL text from the full image + best YOLO crop.

Why EasyOCR instead of TrOCR?
  • TrOCR is a single-line document model — it collapses the entire image into
    one sequence and hallucinates when it sees pill bubbles / patterns.
  • EasyOCR runs CRAFT text-detection first (finds every text region) then
    decodes each region independently — perfect for busy medicine strips.

Strategy:
  • Use YOLO purely as a "is this a medicine strip?" gate (confidence score).
  • Run EasyOCR on the FULL image so the label panel is never cropped out.
  • Smart scoring picks the medicine name from all detected text blocks.
"""

import re
import cv2
import numpy as np
import easyocr
import torch
from ultralytics import YOLO

# ── MODULE-LEVEL GLOBALS ──────────────────────────────────────────────────────
_yolo_model = None
_ocr_reader = None
_device     = None

# ── Noise words — skip when scoring candidate medicine names ─────────────────
_NOISE_WORDS = {
    "ip", "bp", "usp", "ep", "mg", "ml", "mcg", "iu", "tablet", "tablets",
    "capsule", "capsules", "injection", "syrup", "cream", "gel", "drops",
    "strip", "rx", "only", "for", "use", "not", "sale", "keep", "out",
    "reach", "children", "store", "below", "mfd", "exp", "batch", "no",
    "lot", "mfg", "date", "expiry", "net", "qty", "each", "contains",
    "see", "insert", "patient", "information", "leaflet", "schedule", "h",
    "www", "com", "in", "ltd", "pvt", "pharma", "pharmaceuticals",
    "prescription", "drug", "caution", "registered", "trademark", "from",
    "light", "moisture", "temperature", "exceeding", "directed", "physician",
    "medical", "practitioner", "manufactured", "limited", "industrial",
    "estate", "registered", "salcete", "goa", "verna", "phase", "iiia",
    "permitted", "colours", "titanium", "dioxide", "quinoline", "yellow",
    "dosage", "composition", "coated", "film", "contains", "hydrochloride",
    "protect", "protected", "medicines", "medicines", "all",
}

# Dose/unit pattern (e.g. "5 mg", "500mg", "10 ml")
_DOSE_RE = re.compile(
    r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|ml|g|iu|%|units?)\b', re.IGNORECASE
)

# Known medicine-name fragments for bonus scoring
_MEDICINE_KW = (
    "cetirizine", "levocetirizine", "levo", "amox", "azith", "parace",
    "ibuprofen", "atorva", "metfor", "omepra", "pantop", "cipro", "dolo",
    "cefixime", "montelukast", "salbutamol", "hydrochloride", "dihydro",
    "chloride", "zine", "mycin", "cillin", "cycline", "statin", "prazole",
    "sartan", "pril", "olol", "dipine", "formin",
)


# ── Startup ───────────────────────────────────────────────────────────────────

def load_models():
    """
    Call once when Django starts (from ml/apps.py).
    Loads YOLOv8 strip detector + EasyOCR reader into memory.
    """
    global _yolo_model, _ocr_reader, _device

    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu = (_device == 'cuda')
    print(f"[StripOCR] Loading on device: {_device}")

    _yolo_model = YOLO(
        r"D:\Coding Section\Mediscan\saved_models\strip_detector.pt"
    )
    print("[StripOCR] YOLOv8 strip detector loaded")

    _ocr_reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)
    print("[StripOCR] EasyOCR reader loaded")
    print("[StripOCR] Ready.\n")


# ── OCR helpers ───────────────────────────────────────────────────────────────

def _run_easyocr(image_bgr: np.ndarray, min_conf: float = 0.10) -> list:
    """Run EasyOCR on a BGR image; returns list of {text, conf, bbox} dicts."""
    raw = _ocr_reader.readtext(
        image_bgr,
        detail=1,
        paragraph=False,
        width_ths=0.7,
        height_ths=0.7,
        decoder='beamsearch',
        beamWidth=5,
    )
    results = []
    for (bbox, text, conf) in raw:
        text = text.strip()
        if text and conf >= min_conf:
            results.append({"text": text, "conf": conf, "bbox": bbox})
    return results


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_candidate(text: str) -> float:
    """Score a text fragment on how likely it is to be the medicine name."""
    t = text.strip()
    if len(t) < 3:
        return -99.0

    score  = 0.0
    words  = re.findall(r"[A-Za-z]+", t)
    lower  = t.lower()

    # ── Positive signals ─────────────────────────────────────────────────────
    real_words = [w for w in words if w.lower() not in _NOISE_WORDS and len(w) > 2]
    score += len(real_words) * 1.5

    if _DOSE_RE.search(t):
        score += 3.0

    capitalised = sum(1 for w in words if w and w[0].isupper() and len(w) > 2)
    score += capitalised * 0.8

    # Medicine-name keyword fragments
    for kw in _MEDICINE_KW:
        if kw in lower:
            score += 6.0
            break

    # Bonus: ALL-CAPS short brand name (e.g. "LEVOCETIRIZINE")
    if t.isupper() and 5 < len(t) <= 30:
        score += 2.0

    # ── Negative signals ─────────────────────────────────────────────────────
    digit_ratio = sum(c.isdigit() for c in t) / max(len(t), 1)
    if digit_ratio > 0.5:
        score -= 10.0       # the "888" killer

    if len(t) <= 4:
        score -= 2.0

    if all(w.lower() in _NOISE_WORDS for w in words if w):
        score -= 5.0

    # Pure address / manufacturing info
    for addr_kw in ("ltd", "limited", "pvt", "goa", "phase", "estate",
                    "industrial", "verna", "salcete", "manufactured",
                    "mfg", "lic", "no"):
        if addr_kw in lower.split():
            score -= 3.0
            break

    return score


def _extract_medicine_name(ocr_results: list) -> str:
    """
    Pick the best medicine name from all OCR results.

    Strategy:
      1. Score every fragment; keep those with score > 0.
      2. Start with the highest-scoring block.
      3. Scan for a 'TABLETS IP X mg' / 'Hydrochloride IP' companion block
         that is directly below the brand name and append it to form the
         canonical medicine name (e.g. 'LEVOCETIRIZINE TABLETS IP 5 mg').
    """
    if not ocr_results:
        return ""

    # Attach scores
    scored = [(r, _score_candidate(r["text"])) for r in ocr_results]
    scored.sort(key=lambda x: x[1], reverse=True)

    positive = [(r, s) for r, s in scored if s > 0]

    if not positive:
        best = max(ocr_results, key=lambda r: r["conf"])
        return best["text"]

    best_r, best_s = positive[0]
    name = best_r["text"]

    # Look for a dosage-form companion: "TABLETS IP 5 mg" / "Tablets IP 5 mg"
    _FORM_RE = re.compile(
        r'(tablet|capsule|syrup|injection|cream|gel|drop|solution)s?\s+ip\s+\d',
        re.IGNORECASE,
    )
    # Also look for standalone dose block that follows the brand name
    _DOSE_ONLY_RE = re.compile(
        r'^\d+(?:\.\d+)?\s*(?:mg|mcg|ml|g)$', re.IGNORECASE
    )

    companion = None
    for r, s in scored:
        t = r["text"]
        if t == name:
            continue
        if _FORM_RE.search(t):
            companion = t
            break

    if companion:
        name = name + " " + companion
    else:
        # Fallback: merge top-2 if it scores at least 1.3× the best alone
        if len(positive) >= 2:
            r2, _ = positive[1]
            candidate = name + " " + r2["text"]
            if _score_candidate(candidate) >= best_s * 1.3:
                name = candidate

    return name.strip()


# ── Public API ────────────────────────────────────────────────────────────────

def predict(image_path: str, conf_threshold: float = 0.40) -> dict:
    """
    Main entry point called from Django views.

    Args:
        image_path    : full path to the uploaded image file
        conf_threshold: minimum YOLO confidence (default 0.40)

    Returns:
        {
            "success"       : True / False,
            "medicine_name" : "Levocetirizine Tablets IP 5 mg" / "",
            "all_text"      : ["raw text block 1", ...],
            "confidence"    : 0.838,   # YOLO strip-detection confidence
            "error"         : "" / "<reason>"
        }
    """
    if _yolo_model is None or _ocr_reader is None:
        return {
            "success": False, "medicine_name": "", "all_text": [],
            "confidence": 0,
            "error": "Models not loaded. Call load_models() first.",
        }

    # ── 1. Read image ─────────────────────────────────────────────────────────
    image = cv2.imread(image_path)
    if image is None:
        return {
            "success": False, "medicine_name": "", "all_text": [],
            "confidence": 0, "error": "Could not read image file.",
        }

    # ── 2. YOLOv8 — gate: is this a medicine strip? ───────────────────────────
    yolo_results = _yolo_model.predict(
        source=image, conf=conf_threshold, verbose=False
    )
    boxes = yolo_results[0].boxes

    if boxes is None or len(boxes) == 0:
        return {
            "success": False, "medicine_name": "", "all_text": [],
            "confidence": 0,
            "error": "No medicine strip detected. Try a clearer photo.",
        }

    best_idx  = int(boxes.conf.argmax())
    best_conf = float(boxes.conf[best_idx])

    # ── 3. EasyOCR on FULL image ──────────────────────────────────────────────
    #   YOLO tells us a strip is there; EasyOCR sees the entire label panel.
    #   This prevents missing the text area when YOLO crops to the blister side.
    print("[StripOCR] Running EasyOCR on full image …")
    ocr_results = _run_easyocr(image, min_conf=0.10)

    # Fallback: if full-image gives nothing, try each YOLO bounding box
    if not ocr_results:
        print("[StripOCR] Full-image OCR empty, trying YOLO crops …")
        h, w = image.shape[:2]
        for i in range(len(boxes)):
            cx1, cy1, cx2, cy2 = map(int, boxes.xyxy[i].tolist())
            pad = 10
            cx1 = max(0, cx1 - pad); cy1 = max(0, cy1 - pad)
            cx2 = min(w, cx2 + pad); cy2 = min(h, cy2 + pad)
            crop = image[cy1:cy2, cx1:cx2]
            crop_results = _run_easyocr(crop, min_conf=0.10)
            ocr_results.extend(crop_results)

    all_texts = [r["text"] for r in ocr_results]

    print(f"[StripOCR] OCR found {len(ocr_results)} text blocks:")
    for r in sorted(ocr_results, key=lambda x: -_score_candidate(x["text"]))[:15]:
        print(f"  [{r['conf']:.2f}] score={_score_candidate(r['text']):.1f}  {r['text']!r}")

    if not ocr_results:
        return {
            "success": False, "medicine_name": "", "all_text": [],
            "confidence": best_conf,
            "error": "Strip detected but no text could be read. Try better lighting.",
        }

    # ── 4. Extract medicine name ──────────────────────────────────────────────
    medicine = _extract_medicine_name(ocr_results)

    if not medicine:
        return {
            "success": False, "medicine_name": "", "all_text": all_texts,
            "confidence": best_conf,
            "error": "Strip detected but medicine name could not be identified.",
        }

    return {
        "success"       : True,
        "medicine_name" : medicine,
        "all_text"      : all_texts,
        "confidence"    : round(best_conf, 3),
        "error"         : "",
    }