import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediscan.settings')
django.setup()

from PIL import Image
from scanner.ml.gemini_vision import gemini_vision_ocr_structured, strip_preamble
from scanner.ml.extractor import extract_structured_ocr, extract_dosage_form
from scanner.ml.salt_parser import parse_salt
from scanner.ml.medicine_matcher import match_medicine

img = Image.open('test_images/strip.jpg')
gem_res = gemini_vision_ocr_structured(img)
raw_gem = gem_res.get("raw_text", "")
gemini_text = gem_res.get("plain_text") or strip_preamble(raw_gem)

print("raw_gem:\n", raw_gem)
print("gemini_text:\n", gemini_text)

gemini_extracted = extract_structured_ocr(gemini_text)
print("gemini_extracted before struct:\n", gemini_extracted)

struct = gem_res.get("structured", {})
print("struct:", struct)

if struct.get("brand") and (not gemini_extracted.get("brand") or gemini_extracted.get("brand").lower() != struct["brand"].lower()):
    gemini_extracted["brand"] = struct["brand"]
if struct.get("ingredients"):
    parsed_salts = parse_salt(struct["ingredients"])
    if parsed_salts:
        gemini_extracted["ingredients"] = parsed_salts
if struct.get("dosage_form"):
    gemini_extracted["dosage_form"] = extract_dosage_form(struct["dosage_form"]) or struct["dosage_form"]
if struct.get("manufacturer") and not gemini_extracted.get("manufacturer"):
    gemini_extracted["manufacturer"] = struct["manufacturer"]

print("gemini_extracted after struct:\n", gemini_extracted)

m = match_medicine(gemini_extracted)
print("Match medicine:", m.get("status"), m.get("medicine", {}).get("name"))
print("Confidence:", m.get("confidence_breakdown"))
