"""
tools/test_missed_samples_rotation.py — Test rotation fallback on the 12 missed photos.
Evaluates 0, 180, 90, 270 degrees with Salt-First retrieval and Brand popularity.
"""

import os
import sys
import time
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

MISSED_DIR = PROJECT_ROOT / "tools" / "missed_samples"

TEST_PHOTOS = [
    ("Zyrtec 20 tablets", "huawei cn (27).jpg", "ZYRTEC"),
    ("Brufen 30 tablets", "huawei p30 (109).jpg", "BRUFEN"),
    ("Betaderm 30  gm cream", "huawei p30 (391).jpg", "BETADERM"),
    ("Fucidin 20 g cream", "huawei p30 (522).jpg", "FUCIDIN"),
    ("Fucidin 15 g cream", "huawei p30 (857).jpg", "FUCIDIN"),
    ("Flagyl 20 tablets", "huawei p30 (56).jpg", "FLAGYL"),
    ("Brufen 30 tablets", "huawei p30 (110).jpg", "BRUFEN"),
    ("Brufen 150 ml syrup", "huawei p30 (475).jpg", "BRUFEN"),
    ("Paramol 20 tablets", "huawei p30 (547).jpg", "PARAMOL"),
    ("Betadine 60  g ointment", "huawei p30 (397).jpg", "BETADINE"),
    ("Motilium 40 tablets", "huawei p30 (839).jpg", "MOTILIUM"),
    ("Betadine 120 ml mouth wash", "huawei p30 (720).jpg", "BETADINE"),
]


def evaluate_with_rotation(image_path: Path, rec_a, brand_index):
    img = Image.open(image_path)
    img_exif = ImageOps.exif_transpose(img)

    rotations_data = []
    # Try angles: 0, 180, 90, 270
    angles = [0, 180, 90, 270]
    chosen_angle = 0
    chosen_res = None
    chosen_lines = []

    for angle in angles:
        rot_img = img_exif.rotate(angle, expand=True) if angle != 0 else img_exif
        lines = rec_a.read_lines(rot_img, beams=1)
        res = match_printed_lines(lines, brand_index=brand_index)

        top_cand = res.get("candidates") and res["candidates"][0]
        top_score = top_cand.get("score", 0.0) if top_cand else 0.0
        top_conf = bool(top_cand and top_cand.get("confident"))
        n_matching_tokens = res.get("n_matching_tokens", 0)

        is_confident = top_conf and (n_matching_tokens >= 2)
        rotations_data.append({
            "angle": angle,
            "lines": lines,
            "res": res,
            "top_score": top_score,
            "top_cand": top_cand,
            "is_confident": is_confident,
            "n_tokens": n_matching_tokens,
        })

        if is_confident:
            chosen_angle = angle
            chosen_res = res
            chosen_lines = lines
            break

    if chosen_res is None:
        # Keep the rotation whose best candidate score is highest
        best_rot = max(rotations_data, key=lambda x: x["top_score"])
        chosen_angle = best_rot["angle"]
        chosen_res = best_rot["res"]
        chosen_lines = best_rot["lines"]

    return chosen_angle, chosen_res, chosen_lines, rotations_data


def main():
    print("=" * 75)
    print("EVALUATING ROTATION FALLBACK & SALT-FIRST RETRIEVAL ON 12 MISSED SAMPLES")
    print("=" * 75)

    rec_a = get_printed_recogniser()
    rec_a._ensure_loaded()
    brand_index = get_brand_word_index()

    for idx, (prod, fname, target) in enumerate(TEST_PHOTOS, 1):
        p = MISSED_DIR / prod / fname
        if not p.exists():
            print(f"[{idx:2d}/12] Missing file: {p}")
            continue

        t0 = time.time()
        angle, res, lines, all_rots = evaluate_with_rotation(p, rec_a, brand_index)
        sec = time.time() - t0

        cands = res.get("candidates", [])
        cand_brands = [c.get("brand", "").upper() for c in cands]
        h1 = bool(cand_brands and cand_brands[0] == target)
        h3 = bool(target in cand_brands[:3])

        print(f"\n[{idx:2d}/12] {prod} | File: {fname}", flush=True)
        print(f"      Target: {target}", flush=True)
        print(f"      Chosen Rotation: {angle}° (in {sec:.2f}s, tried {[r['angle'] for r in all_rots]})", flush=True)
        print(f"      Lines Read ({len(lines)}): {' // '.join(lines[:5])}", flush=True)
        print(f"      Top-3 Candidates:", flush=True)
        for rank, c in enumerate(cands[:3], 1):
            marker = " [CORRECT TARGET!]" if c.get("brand", "").upper() == target else ""
            print(f"        {rank}. {c.get('brand')} ({c.get('name')}) — score: {c.get('score')} | type: {c.get('match_type')} | salt_matched: {c.get('salt_matched')}{marker}", flush=True)
        print(f"      Outcome: Top-1={'YES' if h1 else 'NO'} | Top-3={'YES' if h3 else 'NO'} | Status={res.get('status')}", flush=True)


if __name__ == "__main__":
    main()
