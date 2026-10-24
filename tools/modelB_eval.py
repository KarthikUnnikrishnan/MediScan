"""
tools/modelB_eval.py — Ground-truth Evaluation Tool for Model B (Handwritten Prescriptions).

Usage:
    python tools/modelB_eval.py [path_to_truth_csv]

Input:
    A CSV with columns: photo,medicines
    where medicines are brand names separated by ';' typed by the reviewer.
    Example:
        photo,medicines
        1.jpg,Disoxin;Telfast;Ascid
        2.jpg,Montair;Bicozin;Adival

Evaluation Rule:
    A shown medicine is correct if its top-1 candidate name starts with the typed
    brand name (case-insensitive).
    - precision = correct shown medicines / all shown medicines
    - recall = correct medicines found / all correct medicines
    - average seconds per photo
"""

import os
import sys
import csv
import time
import pathlib
from typing import List, Dict, Any

# Bootstrap Django settings
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mediscan.settings")

import django
django.setup()

from PIL import Image
from scanner.ml.recognisers import get_handwritten_recogniser
from scanner.ml import match_word

TEST_DIR = pathlib.Path(r"D:\MediScan_data\ModelB_raw\illegible_rx\data")
DEFAULT_TEMPLATE = pathlib.Path(__file__).parent / "modelB_truth_template.csv"
DEFAULT_TRUTH = pathlib.Path(__file__).parent / "modelB_truth.csv"


def evaluate_photo(recogniser, photo_path: pathlib.Path, expected_brands: List[str]) -> Dict[str, Any]:
    t0 = time.time()
    img = Image.open(photo_path)
    # Run with standard prescription settings (beams=2, min_score=80.0)
    res = recogniser.read_prescription(img, matcher=match_word, beams=2, min_score=80.0)
    elapsed = time.time() - t0

    medicines = res.get("medicines", [])
    # Shown medicines: candidates with score >= 80.0 passing gate
    shown_medicines = [m for m in medicines if m.get("candidates") and m["candidates"][0].get("score", 0) >= 80.0]

    shown_names = [m["candidates"][0]["name"] for m in shown_medicines]

    # Matching rule: shown medicine is correct if top-1 candidate starts with typed brand name (case-insensitive)
    correct_shown = []
    matched_expected = set()

    for m in shown_medicines:
        top_name = m["candidates"][0]["name"]
        matched_brand = None
        for exp in expected_brands:
            if exp and top_name.lower().startswith(exp.lower()):
                matched_brand = exp
                matched_expected.add(exp.lower())
                break
        if matched_brand:
            correct_shown.append({
                "shown": top_name,
                "read": m.get("read", ""),
                "score": m["candidates"][0].get("score", 0),
                "matched_expected": matched_brand
            })

    num_shown = len(shown_medicines)
    num_correct_shown = len(correct_shown)
    num_expected = len([b for b in expected_brands if b])

    if num_shown > 0:
        precision = num_correct_shown / num_shown
    else:
        precision = 1.0 if num_expected == 0 else 0.0

    if num_expected > 0:
        recall = len(matched_expected) / num_expected
    else:
        recall = 1.0 if num_shown == 0 else 0.0

    return {
        "photo": photo_path.name,
        "elapsed": elapsed,
        "timing": res.get("timing", {}),
        "expected": expected_brands,
        "shown_names": shown_names,
        "shown_count": num_shown,
        "correct_count": num_correct_shown,
        "expected_count": num_expected,
        "precision": precision,
        "recall": recall,
        "correct_shown": correct_shown,
        "other_text_count": len(res.get("other_text", []))
    }


def main():
    if len(sys.argv) > 1:
        csv_path = pathlib.Path(sys.argv[1])
    elif DEFAULT_TRUTH.exists():
        csv_path = DEFAULT_TRUTH
    else:
        csv_path = DEFAULT_TEMPLATE

    print("=" * 80)
    print("MediScan Model B — Ground Truth Evaluation")
    print(f"CSV Path   : {csv_path.resolve()}")
    print(f"Photos Dir : {TEST_DIR}")
    print("=" * 80)

    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)

    items = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            photo = row.get("photo", "").strip()
            meds_str = row.get("medicines", "").strip()
            brands = [b.strip() for b in meds_str.split(";") if b.strip()]
            if photo:
                items.append((photo, brands))

    if not items:
        print("No items found in CSV file.")
        sys.exit(1)

    print(f"Found {len(items)} photo entries in CSV.\n")
    print("Loading Model B (HandwrittenTrOCR fp16)...")
    rec = get_handwritten_recogniser()
    rec._ensure_loaded()
    print("Model B ready.\n")

    results = []
    for photo_name, expected in items:
        p_path = TEST_DIR / photo_name
        if not p_path.exists():
            print(f"[SKIP] Photo file not found: {p_path}")
            continue

        print(f"Evaluating {photo_name} (Expected: {expected or 'None specified'})...")
        eval_res = evaluate_photo(rec, p_path, expected)
        results.append(eval_res)

        print(f"  Shown: {eval_res['shown_count']} | Correct: {eval_res['correct_count']} | "
              f"Precision: {eval_res['precision']*100:.1f}% | Recall: {eval_res['recall']*100:.1f}% | "
              f"Time: {eval_res['elapsed']:.2f}s")
        if eval_res["shown_names"]:
            print(f"  Shown meds: {', '.join(eval_res['shown_names'])}")
        print()

    if not results:
        print("No evaluation results to report.")
        return

    # Summary table
    print("=" * 80)
    print("PER-PHOTO EVALUATION RESULTS")
    print("=" * 80)
    header = f"{'Photo':<10} | {'Expected':<10} | {'Shown':<8} | {'Correct':<8} | {'Precision':<10} | {'Recall':<8} | {'Time (s)':<8}"
    print(header)
    print("-" * 80)
    for r in results:
        print(f"{r['photo']:<10} | {r['expected_count']:<10} | {r['shown_count']:<8} | {r['correct_count']:<8} | "
              f"{r['precision']*100:>8.1f}% | {r['recall']*100:>6.1f}% | {r['elapsed']:>7.2f}s")
    print("=" * 80)

    # Totals
    tot_shown = sum(r["shown_count"] for r in results)
    tot_correct = sum(r["correct_count"] for r in results)
    tot_expected = sum(r["expected_count"] for r in results)
    times = [r["elapsed"] for r in results]
    avg_sec = sum(times) / len(times) if times else 0.0

    overall_precision = (tot_correct / tot_shown) if tot_shown > 0 else 0.0
    overall_recall = (tot_correct / tot_expected) if tot_expected > 0 else 0.0

    print("TOTALS & AVERAGES")
    print("-" * 80)
    print(f"Total Photos Evaluated : {len(results)}")
    print(f"Total Correct Expected : {tot_expected}")
    print(f"Total Shown Medicines  : {tot_shown}")
    print(f"Total Correct Shown    : {tot_correct}")
    print(f"Overall Precision      : {overall_precision * 100:.1f}% ({tot_correct}/{tot_shown})")
    print(f"Overall Recall         : {overall_recall * 100:.1f}% ({tot_correct}/{tot_expected})")
    print(f"Average Seconds/Photo  : {avg_sec:.2f}s (target <= 10.0s)")
    print(f"Worst Seconds/Photo    : {max(times):.2f}s (target <= 20.0s)")
    print("=" * 80)


if __name__ == "__main__":
    main()
