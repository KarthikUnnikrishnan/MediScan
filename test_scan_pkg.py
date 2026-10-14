import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediscan.settings')
django.setup()

from PIL import Image
from scanner.ml import scan_medicine_package

img = Image.open('test_images/strip.jpg')
res = scan_medicine_package(img)
print('STATUS:', res.get('status'))
print('MEDICINE:', res.get('medicine', {}).get('name') if res.get('medicine') else None)
print('RAW_OCR_TROCR:', res.get('raw_ocr_trocr'))
print('RAW_OCR_GEMINI:', res.get('raw_ocr_gemini'))
print('CONFIDENCE:', res.get('confidence_breakdown'))
