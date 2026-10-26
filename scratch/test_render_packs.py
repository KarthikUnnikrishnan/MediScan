import os, sys
from pathlib import Path

PROJECT_ROOT = Path(r"d:\Coding Section\MediScan - Original")
sys.path.insert(0, str(PROJECT_ROOT))
os.environ['DJANGO_SETTINGS_MODULE'] = 'mediscan.settings'
import django
django.setup()

from django.template.loader import render_to_string
from scanner.ml.medicine_matcher import match_printed_lines
from scanner.ml.salt_parser import parse_salt
from scanner.ml.alternative_matcher import find_top_cheaper_alternatives
from scanner.ml.drug_safety import get_drug_safety_info

packs = {
    "a) Levocetirizine 5 mg": [
        "LEVOCETIRIZINE TABLETS IP 5 mg",
        "STED BY THE PHYSICIAN",
        "DOSAGE AS DIRECTED BY THE PHYSICIAN",
        "STORE IN A COOL DRY PLACE",
        "KEEP OUT OF REACH OF CHILDREN"
    ],
    "b) Fexomine-180": [
        "FEXOMINE-F",
        "TABLETS IP 180 mg",
        "FEXOFENADINE HYDROCHLORIDE TABLETS IP 180 mg",
        "10 TABLETS"
    ],
    "c) Lorinol-10": [
        "LORINOL-10",
        "LORATADINE TABLETS IP 10 mg",
        "10 TABLETS"
    ]
}

for name, lines in packs.items():
    print("=" * 70)
    print(f"RUNNING: {name}")
    print("=" * 70)
    match_res = match_printed_lines(lines)
    top_cand = match_res.get("medicine") or (match_res.get("candidates") and match_res["candidates"][0])
    result_level = match_res.get("result_level", "likely")
    is_verified = (result_level == "verified")

    verified_med = top_cand
    brand_display = verified_med.get("brand") or (verified_med.get("name", "").split()[0] if verified_med.get("name") else "Generic")
    descriptive_name = verified_med.get("detected_name") or verified_med.get("name", "Medicine")
    verified_med["detected_name"] = descriptive_name
    verified_med["brand"] = brand_display

    alternatives = []
    max_savings = None
    max_savings_percent = None
    if is_verified:
        if not verified_med.get("parsed_salt") and verified_med.get("salt"):
            verified_med["parsed_salt"] = parse_salt(verified_med["salt"])
        alternatives = find_top_cheaper_alternatives(verified_med, limit=5)
        for alt in alternatives:
            s_amt = alt.get("saving_amount")
            if s_amt is not None:
                if max_savings is None or s_amt > max_savings:
                    max_savings = s_amt
                    max_savings_percent = alt.get("saving_percent")

    scan_result = {
        "success": True,
        "status": match_res.get("status"),
        "result_level": result_level,
        "message": match_res.get("message"),
        "strength_mismatch_msg": match_res.get("strength_mismatch_msg", ""),
        "mode": "medicine_package",
        "medicine": verified_med,
        "candidates": [verified_med],
        "top_candidates": [verified_med],
        "medicines": [verified_med],
        "alternatives": alternatives,
        "max_savings": max_savings,
        "max_savings_percent": max_savings_percent,
        "raw_ocr": " // ".join(lines),
    }

    context = {
        "result": scan_result,
        "result_level": result_level,
        "error_reason": "",
        "is_ingredient_match": (top_cand.get("match_type") == "ingredient"),
        "top_candidates": [verified_med],
        "mode": "medicine_package",
        "success": True,
        "medicines": [verified_med],
        "alternatives": alternatives,
        "raw_ocr": " // ".join(lines),
        "detected_name": descriptive_name,
        "per_medicine_results": {},
        "per_medicine_json": "{}",
    }

    html = render_to_string("scanner/result.html", context)

    # Verifications
    has_top_candidates_panel = "TOP CANDIDATES CONSIDERED" in html
    has_score_percent = "% score" in html or "% Score" in html
    has_match_conf = "Match Confidence" in html
    has_trust_conf = "Trust Confidence" in html
    has_sted_ml = "Sted-ML" in html or "sted-ml" in html.lower()

    print(f"Result Level: {result_level}")
    print(f"Detected Medicine: {descriptive_name}")
    print(f"Brand: {verified_med.get('brand')}")
    print(f"Message: {scan_result.get('message')}")
    print(f"Alternatives Count: {len(alternatives)}")
    print(f"Checks:")
    print(f"  - Top candidates panel present? {has_top_candidates_panel} (Must be False)")
    print(f"  - '% score' present?            {has_score_percent} (Must be False)")
    print(f"  - 'Match Confidence' present?  {has_match_conf} (Must be False)")
    print(f"  - 'Trust Confidence' present?  {has_trust_conf} (Must be False)")
    print(f"  - 'Sted-ML' present?           {has_sted_ml} (Must be False)")

    # Save rendered html for inspection
    out_dir = Path("tools/rendered_html")
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = name.split()[0].replace(")", "")
    (out_dir / f"{slug}_single_card.html").write_text(html, encoding="utf-8")
    print(f"Saved HTML to tools/rendered_html/{slug}_single_card.html\n")
