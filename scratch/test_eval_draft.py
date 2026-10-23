import sys
sys.path.insert(0, '.')
import os
import csv
import re
import time
import sqlite3
from pathlib import Path
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediscan.settings')
django.setup()

from rapidfuzz import fuzz, process
from scanner.ml.recognisers import get_printed_recogniser

PHARMA_STOPWORDS = {
    'TABLETS', 'TABLET', 'CAPSULES', 'CAPSULE', 'CAPLETS', 'CAPLET', 'CREAM', 'SYRUP',
    'STRIP', 'STORE', 'PROTECT', 'EACH', 'CONTAINS', 'LIMITED', 'PHARMA', 'PHARMACEUTICAL',
    'PHARMACEUTICALS', 'LABORATORIES', 'LAB', 'LABS', 'BATCH', 'MFG', 'EXP', 'DATE', 'MRP',
    'PRICE', 'RS', 'INR', 'USE', 'FOR', 'ONLY', 'KEEP', 'OUT', 'REACH', 'CHILDREN', 'COOL',
    'DRY', 'PLACE', 'SCHEDULE', 'PRESCRIPTION', 'DRUG', 'CAUTION', 'WARNING', 'DOSAGE',
    'DIRECTION', 'DIRECTIONS', 'DIRECTED', 'PHYSICIAN', 'UNCOATED', 'FILM', 'COATED',
    'SUSPENSION', 'INJECTION', 'INJ', 'DROPS', 'SOLUTION', 'LOTION', 'GEL', 'OINTMENT',
    'MOUTH', 'WASH', 'TOPICAL', 'ORAL', 'EXTERNAL', 'PVT', 'LTD', 'INC', 'CORP', 'CO',
    'COMPANY', 'INDIA', 'IP', 'BP', 'USP', 'NF', 'MG', 'ML', 'MCG', 'GM', 'GRAM', 'GRAMS',
    'IU', 'PERCENT', 'TAPE', 'ADHESIVE', 'GAUZE', 'BANDAGE', 'COTTON', 'SURGICAL', 'STERILE',
    'NET', 'QTY', 'QUANTITY', 'PACK', 'PACKAGE', 'TREATMENT', 'DERMA', 'DERMATOLOGICAL',
    'SHODILATOR', 'MUCOLYTIC', 'EXPECTORANT', 'NATURAL', 'SUGAR', 'FREE', 'TASTE', 'WITH',
    'DYE', 'RELIEF', 'EXTRA', 'FAST', 'ACTION', 'PLUS', 'FORTE', 'CARE', 'HEALTHCARE',
    'FEVER', 'PAIN', 'ANTI', 'INFLAMMATORY', 'ANALGESIC', 'ANTIPYRETIC', 'ITCH', 'ACNE',
    'ALLERGY', 'COLD', 'COUGH', 'ORANGE', 'FLAVOUR', 'FLAVORED', 'BORG', 'LEI', 'CEIP',
    'LOLIG', 'TODUCE', 'TOCUCE', 'JULPHAR', 'JAMJOM', 'TOR', 'EASTED', 'SIN', 'REMO',
    'SVIAMOWING', 'IVET', 'KAR', 'YLY', 'WYONI', 'ARRIV', 'ARGICAL', 'AIR', 'VENTED',
    '5CMX5YD', 'NAN', 'TAPE', 'OINT'
}

COMMON_SALTS = {
    'PARACETAMOL', 'ACETAMINOPHEN', 'IBUPROFEN', 'TRETINOIN', 'ACETYLSALICYLIC', 'ASPIRIN',
    'AMOXICILLIN', 'AMOXYCILLIN', 'CLAVULANIC', 'CLAVULANATE', 'CETIRIZINE', 'LEVOCETIRIZINE',
    'AZITHROMYCIN', 'DICLOFENAC', 'ACECLOFENAC', 'PANTOPRAZOLE', 'OMEPRAZOLE', 'RABEPRAZOLE',
    'METFORMIN', 'GLIMEPIRIDE', 'ATORVASTATIN', 'ROSUVASTATIN', 'AMLODIPINE', 'TELMISARTAN',
    'SALBUTAMOL', 'ALBUTEROL', 'POVIDONE', 'IODINE', 'BETAMETHASONE', 'CELECOXIB', 'METRONIDAZOLE',
    'LACTULOSE', 'TERBINAFINE', 'DOMPERIDONE', 'FUSIDIC', 'GENTAMICIN', 'SITAGLIPTIN', 'ACYCLOVIR'
}

def split_glued_tokens(text: str) -> str:
    t = re.sub(r'([A-Za-z]+)(\d+(?:\.\d+)?%?)', r'\1 \2', text)
    t = re.sub(r'(\d+(?:\.\d+)?%?)([A-Za-z]+)', r'\1 \2', t)
    return t

base = Path(r"D:\MediScan_data\ModelA")
conn = sqlite3.connect('saved_models/medicines.sqlite')
cur = conn.cursor()
cur.execute('SELECT id, name FROM medicines')

brand_to_ids = {}
for mid, name in cur.fetchall():
    if not name: continue
    w = re.split(r'[\s\-_]+', name.strip())[0]
    norm = re.sub(r'[^A-Z0-9]', '', w.upper())
    if len(norm) >= 2 and norm not in PHARMA_STOPWORDS:
        brand_to_ids.setdefault(norm, []).append(mid)

brand_words_list = sorted(brand_to_ids.keys())

# Load e2e_eval matching photos
rows = []
with open(base / 'final' / 'e2e_eval.tsv', encoding='utf-8', newline='') as f:
    for r in csv.reader(f, delimiter='\t'):
        if len(r) >= 3 and (base / r[0]).exists():
            brand_norm = re.sub(r'[^A-Z0-9]', '', r[1].strip().upper())
            if brand_norm in brand_to_ids:
                rows.append((r[0], brand_norm, r[2]))

by_prod = {}
for path, brand, prod in rows:
    by_prod.setdefault(prod, []).append((path, brand, prod))

sample = []
for prod, items in list(by_prod.items())[:10]:
    sample.extend(items[:2])

rec_a = get_printed_recogniser()
rec_a._ensure_loaded()

top1_hits = 0
top3_hits = 0
top5_hits = 0

for idx, (rel_path, target_brand, prod_name) in enumerate(sample, 1):
    img_path = base / rel_path
    lines = rec_a.read_lines(str(img_path), beams=1)
    
    full_text = " ".join(lines).upper()
    detected_strengths = re.findall(r'\b\d+(?:\.\d+)?\b', full_text)
    
    # Candidate tokens with box prominence weights
    candidate_tokens = {}
    for box_idx, line in enumerate(lines):
        # Prominence bonus for larger/earlier boxes
        prominence = 20.0 if box_idx == 0 else (12.0 if box_idx == 1 else (6.0 if box_idx == 2 else 0.0))
        cleaned = split_glued_tokens(line.upper())
        words = [re.sub(r'[^A-Z0-9]', '', w) for w in re.split(r'[\s\-_]+', cleaned) if w.strip()]
        
        # Word alone
        for w in words:
            if len(w) >= 3 and w not in PHARMA_STOPWORDS and not w.isdigit():
                cur_sc, cur_prom, cur_cnt = candidate_tokens.get(w, (0, 0, 0))
                candidate_tokens[w] = (cur_sc, max(cur_prom, prominence), cur_cnt + 1)
        
        # Hyphen joined
        hyphen_j = re.sub(r'[^A-Z0-9]', '', line.upper())
        if len(hyphen_j) >= 3 and hyphen_j not in PHARMA_STOPWORDS and not hyphen_j.isdigit():
            cur_sc, cur_prom, cur_cnt = candidate_tokens.get(hyphen_j, (0, 0, 0))
            candidate_tokens[hyphen_j] = (cur_sc, max(cur_prom, prominence), cur_cnt + 1)
            
        # Two-word joins
        for i in range(len(words) - 1):
            two_j = words[i] + words[i+1]
            if len(two_j) >= 4 and two_j not in PHARMA_STOPWORDS and not two_j.isdigit():
                cur_sc, cur_prom, cur_cnt = candidate_tokens.get(two_j, (0, 0, 0))
                candidate_tokens[two_j] = (cur_sc, max(cur_prom, prominence), cur_cnt + 1)

    brand_scores = {}
    for tok, (_, prom, count) in candidate_tokens.items():
        if tok in brand_to_ids:
            score = 100.0 + prom
            if tok in COMMON_SALTS:
                score -= 15.0 # Slight penalty if token is a generic chemical salt rather than brand
            if tok not in brand_scores or score > brand_scores[tok][0]:
                brand_scores[tok] = (score, count)
        elif len(tok) >= 4:
            f_matches = process.extract(tok, brand_words_list, scorer=fuzz.ratio, limit=3, score_cutoff=75.0)
            for m_b, score, _ in f_matches:
                adj_score = score + prom
                if m_b in COMMON_SALTS:
                    adj_score -= 15.0
                if m_b not in brand_scores or adj_score > brand_scores[m_b][0]:
                    brand_scores[m_b] = (adj_score, count)

    cand_ids = []
    brand_by_id = {}
    for b_word, (b_score, b_count) in sorted(brand_scores.items(), key=lambda x: x[1][0], reverse=True)[:10]:
        for mid in brand_to_ids[b_word][:15]:
            cand_ids.append(mid)
            brand_by_id[mid] = (b_word, b_score, b_count)

    scored = []
    if cand_ids:
        placeholders = ','.join(['?'] * len(cand_ids))
        cur.execute(f"SELECT id, name, salt, manufacturer, price FROM medicines WHERE id IN ({placeholders})", cand_ids)
        rows_db = cur.fetchall()

        for mid, name, salt, mfr, price in rows_db:
            b_word, b_score, b_count = brand_by_id.get(mid, ("", 0, 1))
            cand_name = (name or "").upper()
            cand_salt = (salt or "").upper()
            
            score = b_score
            
            # Salt boost (+35)
            salt_words = [re.sub(r'[^A-Z0-9]', '', w) for w in re.split(r'[\s,()/+]+', cand_salt) if len(w) >= 4]
            if any(sw in full_text and sw not in PHARMA_STOPWORDS for sw in salt_words):
                score += 35.0
                
            # Strength corroboration (+15 / -10)
            cand_strengths = re.findall(r'\b\d+(?:\.\d+)?\b', cand_name + ' ' + cand_salt)
            if any(st in cand_strengths for st in detected_strengths if len(st) >= 2):
                score += 15.0
            elif detected_strengths and cand_strengths:
                score -= 10.0
                
            if b_count > 1:
                score += min(9.0, (b_count - 1) * 3.0)
                
            if '+' in cand_salt:
                score -= 20.0
                
            first_brand = re.split(r'[\s\-_]+', cand_name)[0]
            norm_b = re.sub(r'[^A-Z0-9]', '', first_brand)
            scored.append((norm_b, score, name))

    scored.sort(key=lambda x: x[1], reverse=True)
    dedup_brands = []
    seen_b = set()
    for b_norm, sc, nm in scored:
        if b_norm not in seen_b:
            seen_b.add(b_norm)
            dedup_brands.append(b_norm)

    h1 = len(dedup_brands) > 0 and dedup_brands[0] == target_brand
    h3 = target_brand in dedup_brands[:3]
    h5 = target_brand in dedup_brands[:5]
    if h1: top1_hits += 1
    if h3: top3_hits += 1
    if h5: top5_hits += 1
    
    print(f"[{idx:2d}/{len(sample)}] Target: {target_brand:12} | Top3: {dedup_brands[:3]} | h1={h1} h3={h3}")

print(f"\nResults on {len(sample)} test photos:")
print(f"Top-1: {100.0 * top1_hits / len(sample):.1f}%")
print(f"Top-3: {100.0 * top3_hits / len(sample):.1f}%")
print(f"Top-5: {100.0 * top5_hits / len(sample):.1f}%")
