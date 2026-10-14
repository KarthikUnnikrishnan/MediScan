import os, sys
sys.path.insert(0, os.path.abspath('.'))
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediscan.settings')
django.setup()

from PIL import Image
from scanner.ml.vision import crop_strip, run_ocr
from scanner.ml.extractor import extract_structured_ocr
from scanner.ml.medicine_matcher import match_medicine

# The P-650 image
p650_path = 'media/uploads/60cbbde1-fe0b-4fbc-b879-5bbae2125090.jpg'
if not os.path.exists(p650_path):
    print("Finding P-650 image...")
    from pathlib import Path
    for p in Path('media/uploads').glob('*.jpg'):
        print(p, p.stat().st_size)

img = Image.open(p650_path)
crop, yolo_hit = crop_strip(img)
print(f"YOLO hit: {yolo_hit}, Crop size: {crop.size}")

trocr_text = run_ocr(crop)
print(f"TrOCR text from crop:\n{repr(trocr_text)}")

trocr_full = run_ocr(img)
print(f"TrOCR text from full image:\n{repr(trocr_full)}")

ext = extract_structured_ocr(trocr_text or trocr_full)
print(f"Extracted from TrOCR:\n{ext}")

match = match_medicine(ext)
print(f"Match from TrOCR:\n{match.get('status')}: {match.get('medicine', {}).get('name') if match.get('medicine') else None}")
print(f"Candidate count: {match.get('candidate_count')}")
