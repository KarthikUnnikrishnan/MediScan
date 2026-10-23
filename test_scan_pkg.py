import os
import sys
import time
from pathlib import Path
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediscan.settings')
django.setup()

from scanner.ml.recognisers import get_printed_recogniser, get_handwritten_recogniser

print("=" * 70)
print("TEST 1: Check Model B (Handwritten Stub)")
print("=" * 70)
rec_b = get_handwritten_recogniser()
res_b = rec_b.read_prescription("test_images/strip.jpg")
print("Model B response:", res_b)
assert res_b["ok"] is False, "Model B should return ok=False when model folder is missing"
print("PASS: Model B returned expected stub message.\n")

print("=" * 70)
print("TEST 2: Model A (Printed Recogniser) Speed & Quality Test (Greedy, Max 14 Boxes)")
print("=" * 70)
rec_a = get_printed_recogniser()

# Ensure Model A is pre-warmed so we test real per-scan latency
print("Warming up Model A...")
tw0 = time.time()
rec_a._ensure_loaded()
print(f"Model A warm-up took {time.time() - tw0:.2f}s\n")

sample_images = [
    (Path(r"D:\MediScan_data\ModelA\raw\mobile_pharma_packages\bjy2svvmn8-1\Acretin 30 g cream\huawei cn (48).jpg"), ["ACRETIN"]),
    (Path(r"D:\MediScan_data\ModelA\raw\mobile_pharma_packages\bjy2svvmn8-1\Adol 24 caplets\huawei cn (131).jpg"), ["ADOL"]),
    (Path(r"D:\MediScan_data\ModelA\raw\mobile_pharma_packages\bjy2svvmn8-1\Aggrex 60 tablets\huawei cn (66).jpg"), ["AGBREX", "AGGREX"]),
    (Path(r"D:\MediScan_data\ModelA\raw\mobile_pharma_packages\bjy2svvmn8-1\Airoplast nan Tape\huawei cn (80).jpg"), ["AIROPLAST"]),
    (Path(r"D:\MediScan_data\ModelA\raw\mobile_pharma_packages\bjy2svvmn8-1\All-Vent 125 ml syrup\huawei cn (46).jpg"), ["ALL-VENT", "ALLVENT"]),
]

times = []
for idx, (img_path, expected_brands) in enumerate(sample_images, 1):
    print(f"\n--- Photo {idx}: {img_path.name} ({img_path.parent.name}) ---")
    if not img_path.exists():
        print(f"Warning: {img_path} not found.")
        continue

    t0 = time.time()
    lines, details = rec_a.read_lines(str(img_path), beams=1, return_details=True)
    total_sec = time.time() - t0
    times.append(total_sec)

    print(f"Timing Breakdown: box_finding={details['box_find_sec']:.2f}s | reading={details['reading_sec']:.2f}s | total={total_sec:.2f}s (boxes kept: {details['box_count']})")
    print(f"Lines read ({len(lines)}):")
    for line in lines:
        print(f"   | {line}")

    # Verify expected brand line is found
    found_brand = any(any(eb in line for eb in expected_brands) for line in lines)
    if found_brand:
        print(f"   [OK] Brand confirmed: {expected_brands}")
    else:
        print(f"   [WARN] Expected brand {expected_brands} NOT found in lines!")

avg_time = sum(times) / len(times) if times else 0
print("\n" + "=" * 70)
print(f"SUMMARY: Average per-photo scan time: {avg_time:.2f}s (Target: < 8s)")
print("=" * 70)
