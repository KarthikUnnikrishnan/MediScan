import os, sys
sys.path.insert(0, os.path.abspath('.'))
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediscan.settings')
django.setup()

from scanner.ml.medicine_matcher import match_medicine
from scanner.ml.salt_parser import parse_salt
import sqlite3

# Test 1: P-650
p650_extracted = {
    'brand': 'P-650',
    'ingredients': parse_salt('Paracetamol (650mg)'),
    'manufacturer': 'apex laboratories',
    'dosage_form': 'Tablet',
    'normalized_ocr': 'Paracetamol Tablets IP P-650 Tablets apex laboratories private limited Paracetamol IP 650 mg',
}
res_p650 = match_medicine(p650_extracted)
print("P-650 match result:")
print("Status:", res_p650.get("status"))
if res_p650.get("medicine"):
    print("Matched medicine name:", res_p650["medicine"]["name"])
    print("Matched medicine ID:", res_p650["medicine"]["id"])
    print("Matched medicine MFR:", res_p650["medicine"].get("manufacturer"))
    print("Breakdown:", res_p650.get("confidence_breakdown"))

# Check if P-650 is in the database!
conn = sqlite3.connect('db.sqlite3')
cur = conn.cursor()
cur.execute("SELECT id, name, salt, manufacturer, price FROM medicines WHERE name LIKE '%P-650%' OR name LIKE '%P 650%' OR name LIKE '%P650%'")
rows = cur.fetchall()
print("\nDatabase rows for P-650:")
for r in rows:
    print(r)

# Test 2: 1-AL
al_extracted = {
    'brand': '1-AL',
    'ingredients': parse_salt('Levocetirizine Hydrochloride IP (5mg)'),
    'manufacturer': 'FDC Limited',
    'dosage_form': 'Tablet',
    'normalized_ocr': '1-AL Levocetirizine Tablets IP 5mg FDC Limited',
}
res_al = match_medicine(al_extracted)
print("\n1-AL match result:")
print("Status:", res_al.get("status"))
if res_al.get("medicine"):
    print("Matched medicine name:", res_al["medicine"]["name"])
    print("Matched medicine ID:", res_al["medicine"]["id"])
    print("Matched medicine MFR:", res_al["medicine"].get("manufacturer"))
    print("Breakdown:", res_al.get("confidence_breakdown"))

cur.execute("SELECT id, name, salt, manufacturer, price FROM medicines WHERE name LIKE '%1-AL%' OR name LIKE '%1 AL%' OR name LIKE '%1AL%'")
rows = cur.fetchall()
print("\nDatabase rows for 1-AL:")
for r in rows:
    print(r)
