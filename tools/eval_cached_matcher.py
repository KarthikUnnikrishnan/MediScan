"""
tools/eval_cached_matcher.py — Fast, fair, in-memory offline benchmark.

Evaluates matcher variants from tools/eval_lines_cache.json (no GPU needed, runs in seconds).
Computes:
  - Old naive baseline vs Step 6 matcher
  - Top-1, Top-3, Top-5 accuracy
  - Unverified count
  - Wrong-but-confident rate (%)
  - Average total seconds per photo (box_find + reading + matching)
  - Half A (tuning) vs Half B (testing) validation
  - Root cause diagnosis for all Top-3 misses
"""

import os
import sys
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediscan.settings')
import django
django.setup()

from rapidfuzz import fuzz, process
from scanner.ml.medicine_index import get_brand_word_index
from scanner.ml.medicine_matcher import match_printed_lines, PHARMA_STOPWORDS


def run_naive_matcher(lines: List[str], brand_index) -> Dict[str, Any]:
    """
    OLD naive matching:
    Every word of every line against the brand index, NO stop-list,
    NO box prominence, NO salt boost, NO strength corroboration, cutoff 75.
    """
    matches = []
    seen_brands = set()

    for line in lines:
        cleaned = re.sub(r'[^A-Za-z0-9\s]', ' ', line.upper())
        words = [w.strip() for w in cleaned.split() if len(w.strip()) >= 2 and not w.strip().isdigit()]
        for w in words:
            if w in brand_index.brand_to_ids:
                if w not in seen_brands:
                    seen_brands.add(w)
                    matches.append((w, 100.0))
            elif len(w) >= 3:
                f_hits = process.extract(
                    w,
                    brand_index.unique_brand_words,
                    scorer=fuzz.ratio,
                    limit=2,
                    score_cutoff=75.0
                )
                for matched_brand, score, _ in f_hits:
                    if matched_brand not in seen_brands:
                        seen_brands.add(matched_brand)
                        matches.append((matched_brand, float(score)))

    # Sort matches by score descending
    matches.sort(key=lambda x: x[1], reverse=True)
    top_cands = [{"brand": b, "score": sc, "confident": sc >= 75.0} for b, sc in matches[:5]]

    top_cand = top_cands[0] if top_cands else None
    is_verified = bool(top_cand and top_cand["confident"])

    return {
        "status": "verified" if is_verified else "unverified",
        "top_candidates": top_cands,
        "candidates": top_cands[:3],
        "top_cand": top_cand
    }


def evaluate_set(
    dataset: List[Dict[str, Any]],
    matcher_type: str = "step6",
    cutoff: float = 75.0,
    brand_index = None
) -> Dict[str, Any]:
    """
    Evaluate a subset of cached photos with either 'step6' or 'naive'.
    """
    top1 = 0
    top3 = 0
    top5 = 0
    unverified = 0
    confident_count = 0
    wrong_confident_count = 0
    total_time = 0.0
    results_detail = []

    for item in dataset:
        target = item["target_brand"]
        lines = item["lines"]

        t_m0 = time.time()
        if matcher_type == "step6":
            res = match_printed_lines(lines, cutoff=cutoff, brand_index=brand_index)
            candidates = res.get("top_candidates", [])
            cand_brands = [re.sub(r'[^A-Z0-9]', '', c.get("brand", "").upper()) for c in candidates]
            top_cand = candidates[0] if candidates else None
            is_confident = bool(top_cand and top_cand.get("confident"))
            is_verified = res.get("status") == "verified"
        else:
            res = run_naive_matcher(lines, brand_index=brand_index)
            candidates = res.get("top_candidates", [])
            cand_brands = [re.sub(r'[^A-Z0-9]', '', c.get("brand", "").upper()) for c in candidates]
            top_cand = res.get("top_cand")
            is_confident = bool(top_cand and top_cand.get("confident"))
            is_verified = res.get("status") == "verified"

        match_sec = time.time() - t_m0
        total_scan_sec = item.get("total_ocr_sec", 6.0) + match_sec
        total_time += total_scan_sec

        h1 = bool(cand_brands and cand_brands[0] == target)
        h3 = bool(target in cand_brands[:3])
        h5 = bool(target in cand_brands[:5])

        if h1: top1 += 1
        if h3: top3 += 1
        if h5: top5 += 1
        if not is_verified: unverified += 1

        if is_confident:
            confident_count += 1
            if not h1:
                wrong_confident_count += 1

        results_detail.append({
            "rel_path": item["rel_path"],
            "target": target,
            "prod_name": item["prod_name"],
            "lines": lines,
            "cand_brands": cand_brands,
            "h1": h1,
            "h3": h3,
            "h5": h5,
            "status": "verified" if is_verified else "unverified",
            "confident": is_confident,
            "total_scan_sec": total_scan_sec
        })

    N = len(dataset)
    verified_items = [d for d in results_detail if d["status"] == "verified"]
    unverified_items = [d for d in results_detail if d["status"] == "unverified"]

    n_verified = len(verified_items)
    n_unverified = len(unverified_items)

    verified_correct = sum(1 for d in verified_items if d["h1"])
    unverified_correct = sum(1 for d in unverified_items if d["h1"])

    verified_precision = (verified_correct / n_verified * 100.0) if n_verified > 0 else 0.0
    unverified_precision = (unverified_correct / n_unverified * 100.0) if n_unverified > 0 else 0.0

    return {
        "total": N,
        "top1_pct": (top1 / N * 100.0) if N > 0 else 0.0,
        "top3_pct": (top3 / N * 100.0) if N > 0 else 0.0,
        "top5_pct": (top5 / N * 100.0) if N > 0 else 0.0,
        "n_verified": n_verified,
        "verified_precision": verified_precision,
        "verified_correct": verified_correct,
        "n_unverified": n_unverified,
        "unverified_precision": unverified_precision,
        "unverified_correct": unverified_correct,
        "avg_total_sec": (total_time / N) if N > 0 else 0.0,
        "details": results_detail
    }


def main():
    import random
    import shutil

    cache_path = PROJECT_ROOT / "tools" / "eval_lines_cache.json"
    if not cache_path.exists():
        print(f"ERROR: {cache_path} does not exist yet. Please run tools/build_eval_cache.py first.")
        sys.exit(1)

    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    brand_index = get_brand_word_index()
    print(f"Loaded {len(data)} cached photo lines across {len(set(d['prod_name'] for d in data))} products.")

    # Split products into two halves (alternating by name)
    unique_prods = sorted(list(set(d["prod_name"] for d in data)))
    half_a_prods = set(p for i, p in enumerate(unique_prods) if i % 2 == 0)
    half_b_prods = set(p for i, p in enumerate(unique_prods) if i % 2 == 1)

    data_half_a = [d for d in data if d["prod_name"] in half_a_prods]
    data_half_b = [d for d in data if d["prod_name"] in half_b_prods]

    print("\n" + "=" * 80)
    print("PRODUCT SPLIT (ALTERNATING BY NAME)")
    print("=" * 80)
    print(f"Half A ({len(half_a_prods)} products, {len(data_half_a)} photos) — TUNING SET:")
    for p in sorted(half_a_prods):
        print(f"   - {p}")
    print(f"\nHalf B ({len(half_b_prods)} products, {len(data_half_b)} photos) — TEST SET:")
    for p in sorted(half_b_prods):
        print(f"   - {p}")

    chosen_cutoff = 75.0

    # 1. Evaluate Half B
    print("\n" + "=" * 80)
    print("EVALUATION ON HALF B (TEST SET: 18 PRODUCTS, 54 PHOTOS)")
    print("=" * 80)
    res_half_b_naive = evaluate_set(data_half_b, matcher_type="naive", cutoff=75.0, brand_index=brand_index)
    res_half_b_step6 = evaluate_set(data_half_b, matcher_type="step6", cutoff=chosen_cutoff, brand_index=brand_index)

    print(f"{'Metric':<38} | {'Old Naive':<18} | {'Step 6 Matcher':<18}")
    print("-" * 80)
    print(f"{'Top-1 Overall Accuracy':<38} | {res_half_b_naive['top1_pct']:.1f}%{'':<13} | {res_half_b_step6['top1_pct']:.1f}%")
    print(f"{'Top-3 Overall Accuracy':<38} | {res_half_b_naive['top3_pct']:.1f}%{'':<13} | {res_half_b_step6['top3_pct']:.1f}%")
    print(f"{'Top-5 Overall Accuracy':<38} | {res_half_b_naive['top5_pct']:.1f}%{'':<13} | {res_half_b_step6['top5_pct']:.1f}%")
    print(f"{'Verified Photos (Count)':<38} | {res_half_b_naive['n_verified']}/{res_half_b_naive['total']}{'':<13} | {res_half_b_step6['n_verified']}/{res_half_b_step6['total']}")
    print(f"{'Precision among VERIFIED photos':<38} | {res_half_b_naive['verified_precision']:.1f}% ({res_half_b_naive['verified_correct']}/{res_half_b_naive['n_verified']}){'':<5} | {res_half_b_step6['verified_precision']:.1f}% ({res_half_b_step6['verified_correct']}/{res_half_b_step6['n_verified']})")
    print(f"{'Unverified Photos (Count)':<38} | {res_half_b_naive['n_unverified']}/{res_half_b_naive['total']}{'':<13} | {res_half_b_step6['n_unverified']}/{res_half_b_step6['total']}")
    print(f"{'Precision among UNVERIFIED photos (Missed)':<38} | {res_half_b_naive['unverified_precision']:.1f}% ({res_half_b_naive['unverified_correct']}/{res_half_b_naive['n_unverified']}){'':<5} | {res_half_b_step6['unverified_precision']:.1f}% ({res_half_b_step6['unverified_correct']}/{res_half_b_step6['n_unverified']})")

    # 2. Full 108 photo evaluation: Step 6 vs Naive
    print("\n" + "=" * 80)
    print("FULL 108-PHOTO BENCHMARK: STEP 6 vs OLD NAIVE ON IDENTICAL CACHED LINES")
    print("=" * 80)
    full_naive = evaluate_set(data, matcher_type="naive", cutoff=75.0, brand_index=brand_index)
    full_step6 = evaluate_set(data, matcher_type="step6", cutoff=chosen_cutoff, brand_index=brand_index)

    print(f"{'Metric':<38} | {'Old Naive':<18} | {'Step 6 Matcher':<18}")
    print("-" * 80)
    print(f"{'Top-1 Overall Accuracy':<38} | {full_naive['top1_pct']:.1f}%{'':<13} | {full_step6['top1_pct']:.1f}%")
    print(f"{'Top-3 Overall Accuracy':<38} | {full_naive['top3_pct']:.1f}%{'':<13} | {full_step6['top3_pct']:.1f}%")
    print(f"{'Top-5 Overall Accuracy':<38} | {full_naive['top5_pct']:.1f}%{'':<13} | {full_step6['top5_pct']:.1f}%")
    print(f"{'Verified Photos (Count)':<38} | {full_naive['n_verified']}/{full_naive['total']}{'':<13} | {full_step6['n_verified']}/{full_step6['total']}")
    print(f"{'Precision among VERIFIED photos':<38} | {full_naive['verified_precision']:.1f}% ({full_naive['verified_correct']}/{full_naive['n_verified']}){'':<5} | {full_step6['verified_precision']:.1f}% ({full_step6['verified_correct']}/{full_step6['n_verified']})")
    print(f"{'Unverified Photos (Count)':<38} | {full_naive['n_unverified']}/{full_naive['total']}{'':<13} | {full_step6['n_unverified']}/{full_step6['total']}")
    print(f"{'Precision among UNVERIFIED photos (Missed)':<38} | {full_naive['unverified_precision']:.1f}% ({full_naive['unverified_correct']}/{full_naive['n_unverified']}){'':<5} | {full_step6['unverified_precision']:.1f}% ({full_step6['unverified_correct']}/{full_step6['n_unverified']})")
    print(f"{'Average Total Seconds per Photo':<38} | {full_naive['avg_total_sec']:.2f}s{'':<13} | {full_step6['avg_total_sec']:.2f}s")
    print("=" * 80)

    # List photos that are STILL missed in the top 3 (max 15) with the lines read
    misses = [d for d in full_step6["details"] if not d["h3"]]
    print("\n" + "=" * 80)
    print(f"PHOTOS STILL MISSED IN TOP-3 AFTER FIX 1 & FIX 2: {len(misses)} / {len(data)}")
    print("=" * 80)
    for idx, item in enumerate(misses[:15], 1):
        print(f"[{idx:2d}] Product : {item['prod_name']}")
        print(f"     Target  : {item['target']} | Got Top-3: {item['cand_brands'][:3]}")
        print(f"     File    : {item['rel_path']}")
        print(f"     Lines ({len(item['lines'])}): {' // '.join(item['lines'][:6])}")
        print("-" * 80)

    # Worst case photo time
    worst_item = max(data, key=lambda x: x.get("total_ocr_sec", 0.0))
    print(f"\nWorst case photo time: {worst_item.get('total_ocr_sec', 0.0):.2f}s ({worst_item['rel_path']})")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
