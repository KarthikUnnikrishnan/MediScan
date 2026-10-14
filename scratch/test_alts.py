import os, sys
sys.path.insert(0, os.path.abspath('.'))
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediscan.settings')
django.setup()

from scanner.ml.alternative_matcher import find_top_cheaper_alternatives

test_med = {
    'id': 1,
    'name': '1-AL 5mg Tablet',
    'salt': 'Levocetirizine (5mg)',
    'price': 63.50,
    'dosage_form': 'Tablet',
    'manufacturer': 'FDC Limited'
}

alts = find_top_cheaper_alternatives(test_med, limit=5)
print(f"Found {len(alts)} alternatives:")
for a in alts:
    print(f"  * {a['name']} by {a['manufacturer']}")
    print(f"    MRP: Rs {a['price']} | Save: Rs {a['saving_amount']} ({a['saving_percent']}%)")
    print(f"    Confidence: {a['confidence']}% ({a['trust_tier']}) | Badge: {a['trust_label']}")
    print(f"    Market Range: {a['current_price_range']} | Reason: {a['trust_reason']}")
