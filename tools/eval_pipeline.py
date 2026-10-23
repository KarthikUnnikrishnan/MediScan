"""
tools/eval_pipeline.py — Comprehensive end-to-end evaluation benchmark.

Reads D:\\MediScan_data\\ModelA\\final\\e2e_eval.tsv, selects photos whose brand word
exists as the first word of a name in medicines.sqlite, takes up to 3 photos per product,
runs the PrintedRecogniser + match_printed_lines pipeline, and reports:
  - brand-in-top-1 %
  - brand-in-top-3 %
  - brand-in-top-5 %
  - average seconds per photo
compared directly to the naive baseline (50.0% top-1, 65.9% top-3).
"""

import os
import sys
import csv
import re
import time
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediscan.settings')
import django
django.setup()

from scanner.ml.recognisers import get_printed_recogniser
from scanner.ml.medicine_index import get_brand_word_index
from scanner.ml.medicine_matcher import match_printed_lines


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate printed medicine packaging OCR & matching pipeline.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(r"D:\MediScan_data\ModelA"),
        help="Base directory containing ModelA dataset."
    )
    parser.add_argument(
        "--tsv-path",
        type=Path,
        default=Path(r"D:\MediScan_data\ModelA\final\e2e_eval.tsv"),
        help="Path to e2e_eval.tsv ground truth file."
    )
    parser.add_argument(
        "--per-product",
        type=int,
        default=3,
        help="Maximum photos to sample per product (default: 3)."
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=None,
        help="Optional maximum total photos to evaluate (useful for smoke testing)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed breakdown for each photo."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("MEDISCAN MODEL A (PRINTED PACKAGING) PIPELINE EVALUATION")
    print("=" * 80)
    print(f"Dataset root : {args.dataset_dir}")
    print(f"TSV Path     : {args.tsv_path}")
    print(f"Per product  : up to {args.per_product} photos")
    if args.max_total:
        print(f"Total cap    : {args.max_total} photos")

    if not args.tsv_path.exists():
        print(f"ERROR: TSV file {args.tsv_path} not found.")
        sys.exit(1)

    print("\n[1/3] Initializing BrandWordIndex...")
    t_idx0 = time.time()
    brand_index = get_brand_word_index()
    print(f"BrandWordIndex loaded ({len(brand_index.unique_brand_words)} words) in {time.time() - t_idx0:.2f}s")

    print("\n[2/3] Filtering evaluation dataset...")
    by_product = {}
    with open(args.tsv_path, encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 3:
                continue
            rel_path, raw_brand, prod_name = row[0], row[1], row[2]
            img_path = args.dataset_dir / rel_path
            if not img_path.exists():
                continue

            brand_norm = re.sub(r'[^A-Z0-9]', '', raw_brand.strip().upper())
            # Only keep photos whose brand exists in brand_word_index
            if brand_norm in brand_index.brand_to_ids:
                by_product.setdefault(prod_name, []).append((img_path, brand_norm, prod_name))

    print(f"Found {len(by_product)} distinct products matching medicines in database.")

    sample_photos = []
    for prod_name in sorted(by_product.keys()):
        items = by_product[prod_name]
        sample_photos.extend(items[:args.per_product])

    if args.max_total and args.max_total < len(sample_photos):
        sample_photos = sample_photos[:args.max_total]

    total_photos = len(sample_photos)
    print(f"Evaluation set selected: {total_photos} photos across {len(set(p[2] for p in sample_photos))} products.")

    print("\n[3/3] Warming up PrintedRecogniser (Model A fp16)...")
    rec_a = get_printed_recogniser()
    t_w0 = time.time()
    rec_a._ensure_loaded()
    print(f"PrintedRecogniser warm-up took {time.time() - t_w0:.2f}s.")

    print("\n" + "-" * 80)
    print("RUNNING BENCHMARK...")
    print("-" * 80)

    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    total_time = 0.0
    times = []

    for idx, (img_path, target_brand, prod_name) in enumerate(sample_photos, 1):
        t0 = time.time()

        # 1. OCR Line Reading (Greedy, top boxes)
        lines, details = rec_a.read_lines(str(img_path), beams=1, return_details=True)

        # 2. Multi-Evidence Matching
        t_match0 = time.time()
        match_result = match_printed_lines(lines, brand_index=brand_index)
        match_sec = time.time() - t_match0

        elapsed = time.time() - t0
        total_time += elapsed
        times.append(elapsed)

        # Candidates evaluation
        candidates = match_result.get("top_candidates", [])
        cand_brands = [re.sub(r'[^A-Z0-9]', '', c.get("brand", "").upper()) for c in candidates]

        h1 = len(cand_brands) > 0 and cand_brands[0] == target_brand
        h3 = target_brand in cand_brands[:3]
        h5 = target_brand in cand_brands[:5]

        if h1: top1_hits += 1
        if h3: top3_hits += 1
        if h5: top5_hits += 1

        top3_display = cand_brands[:3]
        status_flag = "[TOP-1]" if h1 else ("[TOP-3]" if h3 else ("[TOP-5]" if h5 else "[MISS]"))

        print(
            f"[{idx:3d}/{total_photos}] {status_flag:7} Target: {target_brand:12} | "
            f"Top-3: {str(top3_display):28} | "
            f"Time: {elapsed:4.2f}s (box:{details['box_find_sec']:.2f}s read:{details['reading_sec']:.2f}s match:{match_sec:.2f}s)"
        )

        if args.verbose and not h1:
            print(f"      OCR lines ({len(lines)}): {lines[:4]}")
            print(f"      Product: {prod_name}")

    avg_time = total_time / total_photos if total_photos > 0 else 0.0
    top1_pct = (top1_hits / total_photos) * 100.0 if total_photos > 0 else 0.0
    top3_pct = (top3_hits / total_photos) * 100.0 if total_photos > 0 else 0.0
    top5_pct = (top5_hits / total_photos) * 100.0 if total_photos > 0 else 0.0

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Naive Baseline':<18} | {'Step 5 MediScan':<18} | {'Improvement':<12}")
    print("-" * 80)
    diff_t1 = top1_pct - 50.0
    diff_t3 = top3_pct - 65.9
    print(f"{'Top-1 Accuracy':<25} | {'50.0%':<18} | {f'{top1_pct:.1f}%':<18} | {f'{diff_t1:+.1f}%':<12}")
    print(f"{'Top-3 Accuracy':<25} | {'65.9%':<18} | {f'{top3_pct:.1f}%':<18} | {f'{diff_t3:+.1f}%':<12}")
    print(f"{'Top-5 Accuracy':<25} | {'N/A':<18} | {f'{top5_pct:.1f}%':<18} | {'-':<12}")
    print(f"{'Avg Time per Photo':<25} | {'~15.0s (beams=4)':<18} | {f'{avg_time:.2f}s':<18} | {f'{15.0 - avg_time:.2f}s faster':<12}")
    print("=" * 80)


if __name__ == "__main__":
    main()
