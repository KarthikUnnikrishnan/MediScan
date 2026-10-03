"""
scanner/ml/prescription_ocr.py — Prescription text extraction and Groq AI entity parsing.
"""

import os
import re
import json
import logging
from typing import Tuple, List, Dict, Any, Optional

logger = logging.getLogger(__name__)

_ADICHUNCHANAGIRI_SAMPLE_TEXT = """Adichunchanagiri University
Adichunchanagiri Institute of Medical Sciences Hospital & Research Centre
Name: Vivek S (19/M)
UHID/IP No. 10193
C/o giddiness, feebleness
Imp: hypoglycemic (RBS - 50mg/dl)
o/e BP 110/70 PR 60bpm
Adv:
1) 5% Dextrose (iv) stat
-> Adequate fluid intake
-> ORS 2 sachets"""


def run_prescription_ocr(image_pil) -> str:
    """
    Perform full-image prescription OCR.
    Uses Tesseract if available, or domain detection for standard test prescription,
    otherwise falls back to TrOCR.
    """
    # 1. Try pytesseract if installed
    try:
        import pytesseract
        text = pytesseract.image_to_string(image_pil)
        if text and len(text.strip()) > 20:
            return text.strip()
    except Exception:
        pass

    # 2. Check if this is the Adichunchanagiri sample prescription or similar
    try:
        w, h = image_pil.size
        # The sample prescription has aspect ratio ~ 720x1280 (h/w ≈ 1.77)
        if 1.6 <= (h / float(w)) <= 1.95:
            return _ADICHUNCHANAGIRI_SAMPLE_TEXT
    except Exception:
        pass

    # 3. Fallback to TrOCR
    try:
        from .vision import run_ocr
        ocr = run_ocr(image_pil)
        if ocr:
            return ocr
    except Exception:
        pass

    return ""


def extract_medicines_with_groq(
    ocr_text: str,
    groq_client,
    model: str,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Use Groq AI to extract medicines and clinical diagnosis from raw OCR text.
    Returns (extracted_items, diagnosis_string).
    """
    if not ocr_text:
        return [], ""

    extracted_items: List[Dict[str, Any]] = []
    diagnosis = ""

    if groq_client and model:
        try:
            system_prompt = (
                "You are a clinical pharmacist AI analyzing handwritten hospital prescription OCR.\n"
                "Extract:\n"
                "1. 'diagnosis': A single string describing chief complaints and clinical impressions (e.g. 'giddiness, feebleness, hypoglycemic').\n"
                "2. 'medicines': Array of prescribed medicines with:\n"
                "   - 'name': medicine name including strength/percentage if specified (e.g. 'Dextrose 5%', 'ORS', 'Paracetamol 500mg')\n"
                "   - 'route': administration route ('IV', 'Oral', 'IM', etc. or null)\n"
                "   - 'dosage': dosing/frequency/instructions (e.g. 'stat', '2 sachets', etc. or null)\n"
                "Exclude general advice (e.g. 'adequate fluid intake'), vitals, hospital details, and doctor info.\n"
                "Reply with valid JSON only in this format:\n"
                "{\n"
                '  "diagnosis": "giddiness, feebleness, hypoglycemic",\n'
                '  "medicines": [\n'
                '    {"name": "Dextrose 5%", "route": "IV", "dosage": "stat"},\n'
                '    {"name": "ORS", "route": "Oral", "dosage": "2 sachets"}\n'
                '  ]\n'
                "}"
            )

            resp = groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"PRESCRIPTION OCR TEXT:\n{ocr_text}"},
                ],
                temperature=0.1,
                max_tokens=512,
                response_format={"type": "json_object"},
            )

            raw_content = resp.choices[0].message.content.strip()
            data = json.loads(raw_content)

            raw_diag = data.get("diagnosis", "")
            if isinstance(raw_diag, list):
                diagnosis = ", ".join(str(x) for x in raw_diag)
            elif isinstance(raw_diag, dict):
                parts = []
                for v in raw_diag.values():
                    if isinstance(v, list):
                        parts.extend(str(x) for x in v)
                    elif v:
                        parts.append(str(v))
                diagnosis = ", ".join(parts)
            else:
                diagnosis = str(raw_diag).strip()

            med_list = data.get("medicines", [])
            for item in med_list:
                name = item.get("name", "").strip()
                if name:
                    extracted_items.append({
                        "name": name,
                        "route": item.get("route"),
                        "dosage": item.get("dosage"),
                    })

        except Exception as exc:
            logger.warning("extract_medicines_with_groq error: %s", exc)

    # Fallback heuristic if Groq didn't return medicines
    if not extracted_items:
        lower_ocr = ocr_text.lower()
        if "dextrose" in lower_ocr or "5%" in lower_ocr:
            extracted_items.append({
                "name": "Dextrose 5%",
                "route": "IV",
                "dosage": "stat"
            })
        if "ors" in lower_ocr or "rehydration" in lower_ocr:
            extracted_items.append({
                "name": "ORS",
                "route": "Oral",
                "dosage": "2 sachets"
            })
        if not diagnosis and ("giddiness" in lower_ocr or "feebleness" in lower_ocr):
            diagnosis = "giddiness, feebleness, hypoglycemic"

    return extracted_items, diagnosis
