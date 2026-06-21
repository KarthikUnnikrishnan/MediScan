import sys, os
sys.path.insert(0, r"D:\Coding Section\Mediscan")  # ← add this

from ml.generic_finder import load_db, find_generics
from ml.drug_info import load_db as load_drug_db, get_side_effects, check_interactions
# ... rest of the file unchanged

from ml.generic_finder import load_db, find_generics
from ml.drug_info import load_db as load_drug_db, get_side_effects, check_interactions

print("=== Loading databases ===")
load_db()
load_drug_db()

print()
print("=== Test 1: Generic Finder ===")
result = find_generics("Dolo 650")
print("Input    :", result["input_name"])
print("Matched  :", result.get("matched_name"))
print("Salt     :", result.get("salt"))
print("Alternatives:")
for a in result.get("alternatives", []):
    print(f"  {a['name'][:40]:40s}  Rs.{a['price']}")

print()
print("=== Test 2: Side Effects ===")
se = get_side_effects("levocetirizine")
print("Drug:", se["drug_name"], "-> matched:", se.get("matched_name"))
for s in se.get("side_effects", []):
    print(f"  {s['frequency']:10s}  {s['name']}")

print()
print("=== Test 3: Drug Interaction ===")
ddi = check_interactions("Aspirin", "Warfarin")
print("Interacts:", ddi["interacts"])
print("Severity :", ddi["severity"])
print("Info     :", ddi["description"][:120])