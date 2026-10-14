import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediscan.settings')
django.setup()

from scanner.ml.salt_parser import parse_salt
from scanner.ml.extractor import extract_structured_ocr
from scanner.ml.medicine_matcher import match_medicine
from scanner.ml.medicine_index import get_medicine_index

raw_gem = """BRAND: 1-AL
GENERIC: Levocetirizine Tablets IP 5 mg
INGREDIENTS: Levocetirizine Hydrochloride IP 5 mg
STRENGTH: 5 mg
DOSAGE_FORM: Tablet
MANUFACTURER: FDC Limited
SCHEDULE: SCHEDULE H PRESCRIPTION DRUG-CAUTION
RAW_TEXT: Each film coated tablet contains: Levocetirizine Hydrochloride IP 5 mg... 1-AL®"""

gem_ext = extract_structured_ocr(raw_gem)
gem_ext['brand'] = '1-AL'
gem_ext['ingredients'] = parse_salt('Levocetirizine Hydrochloride IP 5 mg')
gem_ext['dosage_form'] = 'Tablet'
gem_ext['manufacturer'] = 'FDC Limited'

print('gem_ext:', gem_ext)

index = get_medicine_index()
print("search_alias('1-AL'):", [m['name'] for m in index.search_alias('1-AL')])

res = match_medicine(gem_ext)
print('Top candidate selected:', res.get('medicine', {}).get('name'))
print('Confidence:', res.get('confidence_breakdown'))
