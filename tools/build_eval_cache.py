"""
tools/build_eval_cache.py — Cache OCR lines for 108 benchmark photos.

Runs PrintedRecogniser once over the 108 photos in e2e_eval.tsv and saves
the recognized lines and timings to tools/eval_lines_cache.json so matcher
experiments and evaluations run in sub-second offline time without GPU inference.
"""

import os
import sys
import csv
import re
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediscan.settings')
import django
django.setup()

from scanner.ml.recognisers import get_printed_recogniser
from scanner.ml.medicine_index import get_brand_word_index

DATASET_DIR = Path(r"D:\MediScan_data\ModelA")
TSV_PATH = DATASET_DIR / "final" / "e2e_eval.tsv"
CACHE_PATH = PROJECT_ROOT / "tools" / "eval_lines_cache.json"
PER_PRODUCT = 3


def main():
    print("=" * 70)
    print("BUILDING OCR LINES CACHE FOR 108 BENCHMARK PHOTOS")
    print("=" * 70)

    brand_index = get_brand_word_index()
    print(f"Loaded BrandWordIndex ({len(brand_index.unique_brand_words)} words)")

    by_product = {}
    with open(TSV_PATH, encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 3:
                continue
            rel_path, raw_brand, prod_name = row[0], row[1], row[2]
            img_path = DATASET_DIR / rel_path
            if not img_path.exists():
                continue

            brand_norm = re.sub(r'[^A-Z0-9]', '', raw_brand.strip().upper())
            if brand_norm in brand_index.brand_to_ids:
                by_product.setdefault(prod_name, []).append({
                    "rel_path": rel_path,
                    "target_brand": brand_norm,
                    "prod_name": prod_name
                })

    sample = []
    for prod_name in sorted(by_product.keys()):
        sample.extend(by_product[prod_name][:PER_PRODUCT])

    print(f"Selected {len(sample)} photos across {len(by_product)} products.")

    rec_a = get_printed_recogniser()
    rec_a._ensure_loaded()
    print("PrintedRecogniser (Model A) warmed up.")

    cached_entries = []
    t_start = time.time()

    for idx, item in enumerate(sample, 1):
        img_path = DATASET_DIR / item["rel_path"]
        t0 = time.time()
        lines, details = rec_a.read_lines(str(img_path), beams=1, return_details=True)
        elapsed = time.time() - t0

        entry = {
            "rel_path": item["rel_path"],
            "target_brand": item["target_brand"],
            "prod_name": item["prod_name"],
            "lines": lines,
            "box_find_sec": details["box_find_sec"],
            "reading_sec": details["reading_sec"],
            "total_ocr_sec": elapsed,
            "box_count": details["box_count"]
        }
        cached_entries.append(entry)

        if idx % 10 == 0 or idx == len(sample):
            print(f"[{idx:3d}/{len(sample)}] Processed: {item['target_brand']:12} | Time: {elapsed:.2f}s | Cumulative: {time.time() - t_start:.1f}s")

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cached_entries, f, indent=2)

    print(f"\nSaved {len(cached_entries)} cached photo lines to {CACHE_PATH}")
    print(f"Total time: {time.time() - t_start:.2f}s (Average: {(time.time() - t_start) / len(cached_entries):.2f}s/photo)")


if __name__ == "__main__":
    main()
