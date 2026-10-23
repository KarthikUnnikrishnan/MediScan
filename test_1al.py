import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediscan.settings')
django.setup()

from scanner.ml.salt_parser import parse_salt
from scanner.ml.extractor import extract_structured_ocr
from scanner.ml.medicine_matcher import match_medicine
from scanner.ml.medicine_index import get_medicine_index

sample_text = """BRAND: 1-AL
GENERIC: Levocetirizine Tablets IP 5 mg
INGREDIENTS: Levocetirizine Hydrochloride IP 5 mg
STRENGTH: 5 mg
DOSAGE_FORM: Tablet
MANUFACTURER: FDC Limited
SCHEDULE: SCHEDULE H PRESCRIPTION DRUG-CAUTION
RAW_TEXT: Each film coated tablet contains: Levocetirizine Hydrochloride IP 5 mg... 1-AL®"""

extracted = extract_structured_ocr(sample_text)
extracted['brand'] = '1-AL'
extracted['ingredients'] = parse_salt('Levocetirizine Hydrochloride IP 5 mg')
extracted['dosage_form'] = 'Tablet'
extracted['manufacturer'] = 'FDC Limited'

print('extracted:', extracted)

index = get_medicine_index()
print("search_alias('1-AL'):", [m['name'] for m in index.search_alias('1-AL')])

res = match_medicine(extracted)
print('Top candidate selected:', res.get('medicine', {}).get('name'))
print('Confidence:', res.get('confidence_breakdown'))
