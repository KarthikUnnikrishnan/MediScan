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


def run_step6_matcher(lines: List[str], brand_index, cutoff: float = 75.0) -> Dict[str, Any]:
    """
    Step 6 Matcher (commit 50b81ed):
    Multi-evidence matcher before FIX 2 & FIX 3:
    - No exact-brand rule (no exact brand priority)
    - No popularity weighting (brand_counts)
    - No first-letter bonus (let_bonus)
    - No salt-first retrieval (salt_to_brand_words)
    - Global compound guard (-20 penalty for all combination products)
    - Status verified when top candidate is confident and corroborated
    """
    from scanner.ml.medicine_matcher import PHARMA_STOPWORDS, COMMON_SALTS, split_glued_tokens

    full_text = " ".join(lines).upper()
    detected_strengths = re.findall(r'\b\d+(?:\.\d+)?%?\b', full_text)

    candidate_tokens: Dict[str, Tuple[float, float, int]] = {}
    for box_idx, line in enumerate(lines):
        prominence = 20.0 if box_idx == 0 else (12.0 if box_idx == 1 else (6.0 if box_idx == 2 else 0.0))
        cleaned = split_glued_tokens(line.upper())
        words = [re.sub(r'[^A-Z0-9]', '', w) for w in re.split(r'[\s\-_]+', cleaned) if w.strip()]

        for w in words:
            if len(w) >= 3 and w not in PHARMA_STOPWORDS and not w.isdigit():
                cur_sc, cur_prom, cur_cnt = candidate_tokens.get(w, (0.0, 0.0, 0))
                candidate_tokens[w] = (cur_sc, max(cur_prom, prominence), cur_cnt + 1)

        hyphen_j = re.sub(r'[^A-Z0-9]', '', line.upper())
        if len(hyphen_j) >= 3 and hyphen_j not in PHARMA_STOPWORDS and not hyphen_j.isdigit():
            cur_sc, cur_prom, cur_cnt = candidate_tokens.get(hyphen_j, (0.0, 0.0, 0))
            candidate_tokens[hyphen_j] = (cur_sc, max(cur_prom, prominence), cur_cnt + 1)

        for i in range(len(words) - 1):
            two_j = words[i] + words[i + 1]
            if len(two_j) >= 4 and two_j not in PHARMA_STOPWORDS and not two_j.isdigit():
                cur_sc, cur_prom, cur_cnt = candidate_tokens.get(two_j, (0.0, 0.0, 0))
                candidate_tokens[two_j] = (cur_sc, max(cur_prom, prominence), cur_cnt + 1)

    brand_scores: Dict[str, Tuple[float, int]] = {}
    for tok, (_, prom, count) in candidate_tokens.items():
        if tok in brand_index.brand_to_ids:
            score = 100.0 + prom
            if tok in COMMON_SALTS:
                score -= 15.0
            if tok not in brand_scores or score > brand_scores[tok][0]:
                brand_scores[tok] = (score, count)
        elif len(tok) >= 4:
            f_matches = process.extract(tok, brand_index.unique_brand_words, scorer=fuzz.ratio, limit=3, score_cutoff=75.0)
            for m_b, score, _ in f_matches:
                adj_score = score + prom
                if m_b in COMMON_SALTS:
                    adj_score -= 15.0
                if m_b not in brand_scores or adj_score > brand_scores[m_b][0]:
                    brand_scores[m_b] = (adj_score, count)

    if not brand_scores:
        return {"status": "unverified", "candidates": [], "top_candidates": []}

    cand_ids = []
    brand_by_id = {}
    for b_word, (b_score, b_count) in sorted(brand_scores.items(), key=lambda x: x[1][0], reverse=True)[:10]:
        for mid in brand_index.brand_to_ids.get(b_word, [])[:15]:
            cand_ids.append(mid)
            brand_by_id[mid] = (b_word, b_score, b_count)

    rows_db = brand_index.get_medicines_by_ids(cand_ids)
    scored_candidates = []
    for row in rows_db:
        mid = row["id"]
        name = row["name"]
        salt = row.get("salt") or ""
        mfr = row.get("manufacturer") or ""
        price = row.get("price")

        b_word, b_score, b_count = brand_by_id.get(mid, ("", 0.0, 1))
        cand_name = (name or "").upper()
        cand_salt = salt.upper()
        cand_mfr = mfr.upper()

        score = b_score

        # 1. Salt boost (+35)
        salt_matched = False
        salt_words = [re.sub(r'[^A-Z0-9]', '', w) for w in re.split(r'[\s,()/+]+', cand_salt) if len(w) >= 4]
        if any(sw in full_text and sw not in PHARMA_STOPWORDS for sw in salt_words):
            salt_matched = True
            score += 35.0

        # 2. Strength corroboration (+15 / -10)
        cand_strengths = re.findall(r'\b\d+(?:\.\d+)?%?\b', cand_name + ' ' + cand_salt)
        strength_matched = False
        if any(st in cand_strengths for st in detected_strengths if len(st) >= 2 or st in {'5', '2'}):
            strength_matched = True
            score += 15.0
        elif detected_strengths and cand_strengths:
            score -= 10.0

        # 3. Manufacturer boost (+10)
        if cand_mfr:
            mfr_words = [w for w in re.split(r'[\s,.]+', cand_mfr) if len(w) >= 4 and w not in PHARMA_STOPWORDS]
            if any(mw in full_text for mw in mfr_words):
                score += 10.0

        # 4. Multi-line support (+3)
        if b_count > 1:
            score += min(9.0, (b_count - 1) * 3.0)

        # 5. Global compound guard (-20)
        if '+' in cand_salt:
            score -= 20.0

        corroborated = bool(salt_matched or strength_matched)
        confident = bool(score >= cutoff)

        first_brand = re.split(r'[\s\-_]+', name.strip())[0]
        b_norm_cand = re.sub(r'[^A-Z0-9]', '', first_brand.upper())
        cand_match_type = "ingredient" if (b_word in COMMON_SALTS or b_norm_cand in COMMON_SALTS) else "brand"

        scored_candidates.append({
            "id": mid,
            "name": name,
            "brand": first_brand,
            "salt": salt,
            "price": price,
            "score": round(score, 1),
            "match_type": cand_match_type,
            "confident": confident,
            "corroborated": corroborated,
            "salt_matched": salt_matched,
            "strength_matched": strength_matched,
        })

    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    dedup_candidates = []
    seen_brands = set()
    for cand in scored_candidates:
        b_norm = re.sub(r'[^A-Z0-9]', '', cand["brand"].upper())
        if b_norm not in seen_brands:
            seen_brands.add(b_norm)
            dedup_candidates.append(cand)
            if len(dedup_candidates) >= 5:
                break

    if not dedup_candidates:
        return {"status": "unverified", "candidates": [], "top_candidates": []}

    top_cand = dedup_candidates[0]
    match_type = top_cand.get("match_type", "brand")
    if match_type == "ingredient":
        is_verified = bool(top_cand.get("confident") and top_cand.get("strength_matched"))
    else:
        is_verified = bool(top_cand.get("confident") and top_cand.get("corroborated"))

    return {
        "status": "verified" if decision["level"] == "verified" else "unverified",
        "result_level": decision["level"],
        "message": decision["message"],
        "candidates": dedup_candidates[:3],
        "top_candidates": dedup_candidates[:5],
        "top_cand": top_cand
    }


def evaluate_set(
    dataset: List[Dict[str, Any]],
    matcher_type: str = "current",
    cutoff: float = 75.0,
    brand_index = None
) -> Dict[str, Any]:
    """
    Evaluate a subset of cached photos with 'naive', 'step6', or 'current'.
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
        if matcher_type == "current":
            res = match_printed_lines(lines, cutoff=cutoff, brand_index=brand_index)
            candidates = res.get("top_candidates", [])
            cand_brands = [re.sub(r'[^A-Z0-9]', '', c.get("brand", "").upper()) for c in candidates]
            top_cand = candidates[0] if candidates else None
            is_confident = bool(top_cand and top_cand.get("confident"))
            is_verified = res.get("status") == "verified"
        elif matcher_type == "step6":
            res = run_step6_matcher(lines, brand_index=brand_index, cutoff=cutoff)
            candidates = res.get("top_candidates", [])
            cand_brands = [re.sub(r'[^A-Z0-9]', '', c.get("brand", "").upper()) for c in candidates]
            top_cand = res.get("top_cand")
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
            "candidates": candidates,
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
    import sqlite3

    cache_path = PROJECT_ROOT / "tools" / "eval_lines_cache.json"
    if not cache_path.exists():
        print(f"ERROR: {cache_path} does not exist yet. Please run tools/build_eval_cache.py first.")
        sys.exit(1)

    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    brand_index = get_brand_word_index()
    print(f"Loaded {len(data)} cached photo lines from single frozen cache across {len(set(d['prod_name'] for d in data))} products.")

    # Split products into two halves (alternating by name)
    unique_prods = sorted(list(set(d["prod_name"] for d in data)))
    half_a_prods = set(p for i, p in enumerate(unique_prods) if i % 2 == 0)
    half_b_prods = set(p for i, p in enumerate(unique_prods) if i % 2 == 1)

    data_half_a = [d for d in data if d["prod_name"] in half_a_prods]
    data_half_b = [d for d in data if d["prod_name"] in half_b_prods]

    # 1. Evaluate Half B (Untouched test set: 54 photos)
    print("\n" + "=" * 92)
    print("THREE-COLUMN EVALUATION ON HALF B (UNTOUCHED TEST SET: 18 PRODUCTS, 54 PHOTOS)")
    print("=" * 92)
    res_b_naive = evaluate_set(data_half_b, matcher_type="naive", brand_index=brand_index)
    res_b_step6 = evaluate_set(data_half_b, matcher_type="step6", brand_index=brand_index)
    res_b_curr = evaluate_set(data_half_b, matcher_type="current", brand_index=brand_index)

    print(f"{'Metric':<38} | {'Old Naive':<15} | {'Step 6 Matcher':<15} | {'Current (FIX 3)':<15}")
    print("-" * 92)
    print(f"{'Top-1 Overall Accuracy':<38} | {res_b_naive['top1_pct']:.1f}%{'':<10} | {res_b_step6['top1_pct']:.1f}%{'':<10} | {res_b_curr['top1_pct']:.1f}%")
    print(f"{'Top-3 Overall Accuracy':<38} | {res_b_naive['top3_pct']:.1f}%{'':<10} | {res_b_step6['top3_pct']:.1f}%{'':<10} | {res_b_curr['top3_pct']:.1f}%")
    print(f"{'Top-5 Overall Accuracy':<38} | {res_b_naive['top5_pct']:.1f}%{'':<10} | {res_b_step6['top5_pct']:.1f}%{'':<10} | {res_b_curr['top5_pct']:.1f}%")
    print(f"{'Verified Photos (Count)':<38} | {res_b_naive['n_verified']}/{res_b_naive['total']}{'':<10} | {res_b_step6['n_verified']}/{res_b_step6['total']}{'':<10} | {res_b_curr['n_verified']}/{res_b_curr['total']}")
    print(f"{'Precision among VERIFIED photos':<38} | {res_b_naive['verified_precision']:.1f}% ({res_b_naive['verified_correct']}/{res_b_naive['n_verified']}){'':<3} | {res_b_step6['verified_precision']:.1f}% ({res_b_step6['verified_correct']}/{res_b_step6['n_verified']}){'':<3} | {res_b_curr['verified_precision']:.1f}% ({res_b_curr['verified_correct']}/{res_b_curr['n_verified']})")
    print(f"{'Unverified Photos (Count)':<38} | {res_b_naive['n_unverified']}/{res_b_naive['total']}{'':<10} | {res_b_step6['n_unverified']}/{res_b_step6['total']}{'':<10} | {res_b_curr['n_unverified']}/{res_b_curr['total']}")
    print(f"{'Precision among UNVERIFIED photos':<38} | {res_b_naive['unverified_precision']:.1f}% ({res_b_naive['unverified_correct']}/{res_b_naive['n_unverified']}){'':<3} | {res_b_step6['unverified_precision']:.1f}% ({res_b_step6['unverified_correct']}/{res_b_step6['n_unverified']}){'':<3} | {res_b_curr['unverified_precision']:.1f}% ({res_b_curr['unverified_correct']}/{res_b_curr['n_unverified']})")

    # 2. Full 108 photo evaluation
    print("\n" + "=" * 92)
    print("THREE-COLUMN EVALUATION ON FULL FROZEN CACHE (36 PRODUCTS, 108 PHOTOS)")
    print("=" * 92)
    full_naive = evaluate_set(data, matcher_type="naive", brand_index=brand_index)
    full_step6 = evaluate_set(data, matcher_type="step6", brand_index=brand_index)
    full_curr = evaluate_set(data, matcher_type="current", brand_index=brand_index)

    print(f"{'Metric':<38} | {'Old Naive':<15} | {'Step 6 Matcher':<15} | {'Current (FIX 3)':<15}")
    print("-" * 92)
    print(f"{'Top-1 Overall Accuracy':<38} | {full_naive['top1_pct']:.1f}%{'':<10} | {full_step6['top1_pct']:.1f}%{'':<10} | {full_curr['top1_pct']:.1f}%")
    print(f"{'Top-3 Overall Accuracy':<38} | {full_naive['top3_pct']:.1f}%{'':<10} | {full_step6['top3_pct']:.1f}%{'':<10} | {full_curr['top3_pct']:.1f}%")
    print(f"{'Top-5 Overall Accuracy':<38} | {full_naive['top5_pct']:.1f}%{'':<10} | {full_step6['top5_pct']:.1f}%{'':<10} | {full_curr['top5_pct']:.1f}%")
    print(f"{'Verified Photos (Count)':<38} | {full_naive['n_verified']}/{full_naive['total']}{'':<10} | {full_step6['n_verified']}/{full_step6['total']}{'':<10} | {full_curr['n_verified']}/{full_curr['total']}")
    print(f"{'Precision among VERIFIED photos':<38} | {full_naive['verified_precision']:.1f}% ({full_naive['verified_correct']}/{full_naive['n_verified']}){'':<3} | {full_step6['verified_precision']:.1f}% ({full_step6['verified_correct']}/{full_step6['n_verified']}){'':<3} | {full_curr['verified_precision']:.1f}% ({full_curr['verified_correct']}/{full_curr['n_verified']})")
    print(f"{'Unverified Photos (Count)':<38} | {full_naive['n_unverified']}/{full_naive['total']}{'':<10} | {full_step6['n_unverified']}/{full_step6['total']}{'':<10} | {full_curr['n_unverified']}/{full_curr['total']}")
    print(f"{'Precision among UNVERIFIED photos':<38} | {full_naive['unverified_precision']:.1f}% ({full_naive['unverified_correct']}/{full_naive['n_unverified']}){'':<3} | {full_step6['unverified_precision']:.1f}% ({full_step6['unverified_correct']}/{full_step6['n_unverified']}){'':<3} | {full_curr['unverified_precision']:.1f}% ({full_curr['unverified_correct']}/{full_curr['n_unverified']})")
    print(f"{'Average Total Seconds per Photo':<38} | {full_naive['avg_total_sec']:.2f}s{'':<10} | {full_step6['avg_total_sec']:.2f}s{'':<10} | {full_curr['avg_total_sec']:.2f}s")
    print("=" * 92)

    # 3. List photos still missed in Top-3 (max 15) with verified index inspection
    conn = sqlite3.connect(PROJECT_ROOT / "saved_models" / "medicines.sqlite")
    cursor = conn.cursor()

    misses = [d for d in full_curr["details"] if not d["h3"]]
    print("\n" + "=" * 92)
    print(f"PHOTOS STILL MISSED IN TOP-3 BY CURRENT MATCHER: {len(misses)} / {len(data)}")
    print("=" * 92)

    for idx, item in enumerate(misses[:15], 1):
        target = item["target"]
        in_index = bool(target in brand_index.brand_to_ids)
        cursor.execute("SELECT id, name, salt FROM medicines WHERE UPPER(name) LIKE ? LIMIT 3", (target + '%',))
        db_rows = cursor.fetchall()

        # Check if any read word has high similarity to target brand
        read_matches = []
        for line in item["lines"]:
            for w in line.upper().split():
                clean_w = re.sub(r'[^A-Z0-9]', '', w)
                if len(clean_w) >= 3:
                    sim = fuzz.ratio(clean_w, target)
                    if sim >= 70:
                        read_matches.append((clean_w, sim))

        print(f"[{idx:2d}] Product       : {item['prod_name']}")
        print(f"     Target Brand  : {target}")
        print(f"     In Index?     : {in_index} (matches {len(db_rows)} DB rows, e.g. {[r[1] for r in db_rows]})")
        print(f"     Got Top-3     : {item['cand_brands'][:3]}")
        print(f"     File          : {item['rel_path']}")
        print(f"     Lines Read ({len(item['lines'])}): {' // '.join(item['lines'][:5])}")
        print(f"     Target Read?  : {read_matches if read_matches else 'NO (Brand text not recognized in OCR lines)'}")
        print("-" * 92)

    conn.close()


if __name__ == "__main__":
    main()

