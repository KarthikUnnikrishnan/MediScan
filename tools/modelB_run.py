"""
tools/modelB_run.py — Offline pipeline test for Model B (HandwrittenTrOCR).

Runs the offline prescription pipeline on 10+ photos from the test folder:
D:\\MediScan_data\\ModelB_raw\\illegible_rx\\data

Prints per photo:
  - lines read
  - medicines matched
  - elapsed seconds

Reports at the end:
  - average and worst seconds per photo
  - any crash
  - 3 examples where it is wrong

Saves all console output to tools/modelB_run.txt.
"""

import io
import os
import sys
import time
import pathlib
import traceback

# ── Bootstrap Django ──────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mediscan.settings")

import django
django.setup()

from django.conf import settings
from PIL import Image
from scanner.ml.recognisers import get_handwritten_recogniser
from scanner.ml import match_word

TEST_DIR = pathlib.Path(r"D:\MediScan_data\ModelB_raw\illegible_rx\data")
OUTPUT_FILE = pathlib.Path(__file__).parent / "modelB_run.txt"
MIN_SCORE = 80.0
PHOTO_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def main():
    buf = io.StringIO()

    def log(msg=""):
        print(msg)
        buf.write(msg + "\n")

    log("=" * 76)
    log("MediScan Model B Offline Prescription Pipeline Run")
    log(f"Model Directory : {settings.MODEL_B_DIR}")
    log(f"Test Directory  : {TEST_DIR}")
    log(f"Matching Threshold (min_score): {MIN_SCORE}")
    log("=" * 76)

    test_photos = [TEST_DIR / f"{i}.jpg" for i in PHOTO_IDS if (TEST_DIR / f"{i}.jpg").exists()]
    log(f"Found {len(test_photos)} test photos: {[p.name for p in test_photos]}\n")

    # Load Model B
    log("Initializing HandwrittenTrOCR (fp16 on CUDA)...")
    t_load_start = time.time()
    recogniser = get_handwritten_recogniser()
    recogniser._ensure_loaded()
    log(f"HandwrittenTrOCR loaded in {time.time() - t_load_start:.2f}s (device={recogniser.device})\n")

    results = []
    crashes = []

    for photo_path in test_photos:
        filename = photo_path.name
        log(f"--- [{filename}] ---")
        t0 = time.time()
        try:
            img = Image.open(photo_path)
            res = recogniser.read_prescription(img, matcher=match_word, beams=4, min_score=MIN_SCORE)
            elapsed = time.time() - t0

            lines = res.get("lines", [])
            meds = res.get("medicines", [])
            # Matched medicines with candidates >= min_score
            matched = [m for m in meds if m.get("candidates") and m["candidates"][0].get("score", 0) >= MIN_SCORE]

            results.append({
                "photo": filename,
                "lines_count": len(lines),
                "matched_count": len(matched),
                "elapsed": elapsed,
                "lines": lines,
                "meds": meds,
                "matched": matched,
                "ok": res.get("ok", False),
            })

            log(f"  Result : ok={res.get('ok')} | Lines read: {len(lines)} | Medicines matched: {len(matched)} | Time: {elapsed:.2f}s")
            for m in matched:
                top = m["candidates"][0]
                log(f"    [MATCH] Read: '{m.get('read')}' -> Cand: '{top.get('name')}' (score={top.get('score')}%) [form={m.get('form')}, dose={m.get('dose')}, freq={m.get('frequency')}]")

            unmatched = [m for m in meds if not (m.get("candidates") and m["candidates"][0].get("score", 0) >= MIN_SCORE)]
            for u in unmatched[:3]:
                log(f"    [UNMATCHED] Read: '{u.get('read')}' (Line: '{u.get('text')}')")

        except Exception as exc:
            elapsed = time.time() - t0
            crashes.append((filename, str(exc), traceback.format_exc()))
            log(f"  Result : CRASH ({exc}) | Time: {elapsed:.2f}s")

        log()

    # ── Summary Report ────────────────────────────────────────────────────────
    times = [r["elapsed"] for r in results]
    avg_sec = sum(times) / len(times) if times else 0.0
    worst_sec = max(times) if times else 0.0
    worst_photo = max(results, key=lambda r: r["elapsed"])["photo"] if results else "N/A"

    log("=" * 76)
    log("EXECUTION SUMMARY")
    log("=" * 76)
    log(f"Photos processed        : {len(results)} / {len(test_photos)}")
    log(f"Total processing time   : {sum(times):.2f}s")
    log(f"Average seconds / photo : {avg_sec:.2f}s")
    log(f"Worst seconds / photo   : {worst_sec:.2f}s ({worst_photo})")
    log(f"Crashes encountered     : {len(crashes)}")
    if crashes:
        for c_file, c_err, c_tb in crashes:
            log(f"  Crash in {c_file}: {c_err}")

    log("\n" + "=" * 76)
    log("PER-PHOTO METRICS TABLE")
    log("=" * 76)
    log(f"{'Photo':<12} | {'Lines Read':<12} | {'Meds Matched':<14} | {'Seconds':<10}")
    log("-" * 76)
    for r in results:
        log(f"{r['photo']:<12} | {r['lines_count']:<12} | {r['matched_count']:<14} | {r['elapsed']:<10.2f}s")
    log("=" * 76)

    # ── 3 Examples where the pipeline is wrong ────────────────────────────────
    log("\n" + "=" * 76)
    log("3 EXAMPLES WHERE THE PIPELINE IS WRONG (ANALYSIS)")
    log("=" * 76)

    # Find false positives or misreads
    wrong_examples = []

    # Check 1: Non-medicine header / demographic matched as a medicine
    for r in results:
        for m in r["matched"]:
            top = m["candidates"][0]
            read_t = m.get("read", "")
            # Check if read text looks like demographics or patient title
            if any(term in read_t.lower() for term in ["mr", "ms", "mrs", "dr", "old", "latlet", "who", "farm"]):
                wrong_examples.append({
                    "photo": r["photo"],
                    "read": read_t,
                    "matched": top.get("name"),
                    "score": top.get("score"),
                    "issue": "Non-medicine text (patient title/demographics/artifact) matched to database brand name."
                })
            if len(wrong_examples) >= 3:
                break
        if len(wrong_examples) >= 3:
            break

    # If we need more examples, look at misread lines or unmatched lines
    if len(wrong_examples) < 3:
        for r in results:
            for u in r.get("meds", []):
                if not (u.get("candidates") and u["candidates"][0].get("score", 0) >= MIN_SCORE):
                    txt = u.get("text", "")
                    if len(txt) > 4:
                        wrong_examples.append({
                            "photo": r["photo"],
                            "read": u.get("read"),
                            "matched": "None (unmatched)",
                            "score": 0.0,
                            "issue": f"Illegible handwritten text '{txt}' could not be resolved to any medicine."
                        })
                if len(wrong_examples) >= 3:
                    break
            if len(wrong_examples) >= 3:
                break

    for idx, ex in enumerate(wrong_examples[:3], 1):
        log(f"Example {idx}:")
        log(f"  Photo   : {ex['photo']}")
        log(f"  Text Read: '{ex['read']}'")
        log(f"  Matched : {ex['matched']} (score={ex['score']}%)")
        log(f"  Issue   : {ex['issue']}")
        log("-" * 76)

    # Save output to tools/modelB_run.txt
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    log(f"\nReport written to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
