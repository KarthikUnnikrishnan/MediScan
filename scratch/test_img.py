import os, sys
sys.path.insert(0, os.path.abspath('.'))
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediscan.settings')
django.setup()

from PIL import Image
from scanner.ml import scan_medicine_package
from pathlib import Path

# Find largest recent file in media/uploads
uploads = list(Path('media/uploads').glob('*.jpg'))
uploads.sort(key=lambda p: p.stat().st_mtime, reverse=True)

for p in uploads[:2]:
    print("Testing image:", p)
    img = Image.open(p)
    res = scan_medicine_package(img)
    print("STATUS:", res.get("status"))
    print("DETECTED NAME:", res.get("medicine", {}).get("name") if res.get("medicine") else "None")
    print("RAW_TROCR:", res.get("raw_ocr_trocr"))
    print("RAW_GEMINI:", res.get("raw_ocr_gemini")[:150] if res.get("raw_ocr_gemini") else "None")
    print("CONFIDENCE:", res.get("confidence_breakdown"))
    print("-" * 50)
