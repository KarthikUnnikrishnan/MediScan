"""
tools/build_eval_cache.py — Cache OCR lines for 108 benchmark photos with Rotation Fallback.

Applies PIL.ImageOps.exif_transpose and evaluates rotation fallback (0, 180, 90, 270 degrees).
Saves recognized lines, chosen rotation, timings, and metadata to tools/eval_lines_cache.json.
"""

import os
import sys
import csv
import re
import json
import time
import shutil
from pathlib import Path
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediscan.settings')
import django
django.setup()

from scanner.ml.recognisers import get_printed_recogniser
from scanner.ml.medicine_index import get_brand_word_index
from scanner.ml.medicine_matcher import match_printed_lines

DATASET_DIR = Path(r"D:\MediScan_data\ModelA")
TSV_PATH = DATASET_DIR / "final" / "e2e_eval.tsv"
CACHE_PATH = PROJECT_ROOT / "tools" / "eval_lines_cache.json"
CACHE_BACKUP = PROJECT_ROOT / "tools" / "eval_lines_cache_old.json"
PER_PRODUCT = 3


def main():
    print("=" * 75, flush=True)
    print("BUILDING OCR LINES CACHE FOR 108 BENCHMARK PHOTOS (WITH ROTATION FALLBACK)", flush=True)
    print("=" * 75, flush=True)

    # Backup existing cache if present
    if CACHE_PATH.exists() and not CACHE_BACKUP.exists():
        shutil.copy2(CACHE_PATH, CACHE_BACKUP)
        print(f"Backed up previous cache to {CACHE_BACKUP}", flush=True)

    brand_index = get_brand_word_index()
    print(f"Loaded BrandWordIndex ({len(brand_index.unique_brand_words)} words, {len(brand_index.unique_salt_words)} salt words)", flush=True)

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

    print(f"Selected {len(sample)} photos across {len(by_product)} products.", flush=True)

    rec_a = get_printed_recogniser()
    rec_a._ensure_loaded()
    print("PrintedRecogniser (Model A) warmed up.\n", flush=True)

    cached_entries = []
    t_start = time.time()
    per_photo_times = []

    for idx, item in enumerate(sample, 1):
        img_path = DATASET_DIR / item["rel_path"]
        raw_img = Image.open(img_path)
        img_exif = ImageOps.exif_transpose(raw_img)

        # FIX 1b: Rotation fallback (0, 180, 90, 270)
        # Read at 0 degrees; if not confident (top candidate not confident, or < 2 read tokens >= 90),
        # also try 180, then 90, then 270 degrees. Stop at first confident result;
        # otherwise keep rotation whose best candidate score is highest.
        angles = [0, 180, 90, 270]
        rotations_attempted = []
        rotations_data = []

        chosen_angle = 0
        chosen_lines = []
        chosen_details = {}
        total_ocr_sec = 0.0

        for angle in angles:
            rotations_attempted.append(angle)
            t_rot0 = time.time()
            rot_img = img_exif.rotate(angle, expand=True) if angle != 0 else img_exif
            lines, details = rec_a.read_lines(rot_img, beams=1, return_details=True)
            elapsed_rot = time.time() - t_rot0
            total_ocr_sec += elapsed_rot

            match_res = match_printed_lines(lines, brand_index=brand_index)
            top_cand = match_res.get("candidates") and match_res["candidates"][0]
            top_score = top_cand.get("score", 0.0) if top_cand else 0.0
            top_conf = bool(top_cand and top_cand.get("confident"))
            n_matching_tokens = match_res.get("n_matching_tokens", 0)

            is_confident = top_conf and (n_matching_tokens >= 2)

            rotations_data.append({
                "angle": angle,
                "lines": lines,
                "details": details,
                "top_score": top_score,
                "is_confident": is_confident,
                "n_tokens": n_matching_tokens,
                "status": match_res.get("status")
            })

            if is_confident:
                chosen_angle = angle
                chosen_lines = lines
                chosen_details = details
                break

        if not chosen_lines:
            # Fallback to rotation with highest candidate score
            best_rot = max(rotations_data, key=lambda x: x["top_score"])
            chosen_angle = best_rot["angle"]
            chosen_lines = best_rot["lines"]
            chosen_details = best_rot["details"]

        per_photo_times.append(total_ocr_sec)

        entry = {
            "rel_path": item["rel_path"],
            "target_brand": item["target_brand"],
            "prod_name": item["prod_name"],
            "lines": chosen_lines,
            "rotation": chosen_angle,
            "rotations_attempted": rotations_attempted,
            "box_find_sec": chosen_details.get("box_find_sec", 0.0),
            "reading_sec": chosen_details.get("reading_sec", 0.0),
            "total_ocr_sec": total_ocr_sec,
            "box_count": chosen_details.get("box_count", len(chosen_lines))
        }
        cached_entries.append(entry)

        # Print progress per photo
        rot_str = f"{chosen_angle}° (tried {rotations_attempted})"
        print(
            f"[{idx:3d}/{len(sample)}] {item['target_brand']:12} | Rot: {rot_str:<18} | "
            f"Time: {total_ocr_sec:5.2f}s | Avg: {sum(per_photo_times)/len(per_photo_times):5.2f}s",
            flush=True
        )

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cached_entries, f, indent=2)

    total_time = time.time() - t_start
    avg_time = total_time / len(cached_entries)
    worst_time = max(per_photo_times)

    print("\n" + "=" * 75, flush=True)
    print("CACHE GENERATION COMPLETE", flush=True)
    print(f"Total Photos Processed: {len(cached_entries)}", flush=True)
    print(f"Total Elapsed Time    : {total_time:.2f}s ({total_time/60:.2f} mins)", flush=True)
    print(f"Average Seconds/Photo : {avg_time:.2f}s (Target: < 12s)", flush=True)
    print(f"Worst Case Photo Time : {worst_time:.2f}s", flush=True)
    print(f"Saved to              : {CACHE_PATH}", flush=True)
    print("=" * 75 + "\n", flush=True)


if __name__ == "__main__":
    main()
