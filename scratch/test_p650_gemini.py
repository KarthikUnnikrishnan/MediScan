import os, sys
sys.path.insert(0, os.path.abspath('.'))
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediscan.settings')
django.setup()

from PIL import Image
from scanner.ml.gemini_vision import gemini_vision_ocr_structured
from scanner.ml.extractor import extract_structured_ocr
from scanner.ml.salt_parser import parse_salt
from scanner.ml.medicine_matcher import match_medicine

img = Image.open('media/uploads/60cbbde1-fe0b-4fbc-b879-5bbae2125090.jpg')
res = gemini_vision_ocr_structured(img)
print("Gemini result:")
print("Success:", res.get("success"))
print("Structured:", res.get("structured"))
print("Raw text:\n", res.get("raw_text"))

struct = res.get("structured", {})
ext = extract_structured_ocr(res.get("plain_text", ""))
if struct.get("brand"):
    ext["brand"] = struct["brand"]
if struct.get("ingredients"):
    ext["ingredients"] = parse_salt(struct["ingredients"])
if struct.get("manufacturer"):
    ext["manufacturer"] = struct["manufacturer"]

print("\nExtracted dictionary:\n", ext)
match = match_medicine(ext)
print("\nMatch medicine result:")
print("Status:", match.get("status"))
if match.get("medicine"):
    print("Name:", match["medicine"]["name"])
    print("ID:", match["medicine"]["id"])
    print("Mfr:", match["medicine"].get("manufacturer"))
print("Validation details top candidates:")
if match.get("validation_details"):
    cand = match["validation_details"].get("candidate", {})
    print("Top candidate in validation details:", cand.get("name"), cand.get("manufacturer"))
