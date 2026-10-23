import os
import sys
import re
import time
import sqlite3
from pathlib import Path
from rapidfuzz import fuzz, process

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
    'BORG', 'LEI', 'CEIP', 'LOLIG', 'TODUCE', 'TOCUCE', 'JULPHAR', 'JAMJOM', 'TOR', 'EASTED'
}

def split_glued_tokens(text: str) -> str:
    # letter followed by digit/fraction/percent: ACRETIN0.05% -> ACRETIN 0.05%
    t = re.sub(r'([A-Za-z]+)(\d+(?:\.\d+)?%?)', r'\1 \2', text)
    # digit followed by letter: 500MG -> 500 MG
    t = re.sub(r'(\d+(?:\.\d+)?%?)([A-Za-z]+)', r'\1 \2', t)
    return t

print("Loading brand index from saved_models/medicines.sqlite...")
t0 = time.time()
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
print(f"Loaded {len(brand_words_list)} brand words in {time.time() - t0:.2f}s")

# Test matching on line sets from the 5 photos
sample_test_lines = {
    "Acretin": ['ACRETIN 0.05%', 'TREATMENT', 'TOPICAL', 'CREAM', 'TRETINOIN', 'DERMA', 'ACNE', 'JAMJOM', 'GM', 'PHARMA', '0.05%'],
    "Adol": ['ADOL', 'PARACETAMOL 500 MG', 'CAPLETS', '24', 'JULPHAR'],
    "Aggrex": ['AGBREX', 'KAR', 'ACETYL SALICYLIC', 'TABLETS', '75 MG', 'ACID', '60'],
    "Airoplast": ['AIROPLAST', 'PHARMAPLAST', 'ADHESIVE TAPE', '5 CM X 5YD', 'VENTED', 'AIRO 5045'],
    "All-Vent": ['ALL-VENT', 'PHARMACEUTICAL INDIA', 'MUCOLYTIC', 'SYRUP', 'MENTHOL', '125', 'FREE'],
}

def extract_candidate_tokens(lines):
    tokens = set()
    for line in lines:
        cleaned = split_glued_tokens(line.upper())
        # word alone
        words = [re.sub(r'[^A-Z0-9]', '', w) for w in re.split(r'[\s\-_]+', cleaned) if w.strip()]
        for w in words:
            if len(w) >= 3 and w not in PHARMA_STOPWORDS and not w.isdigit():
                tokens.add(w)
        # hyphen-joined
        hyphen_joined = re.sub(r'[^A-Z0-9]', '', line.upper())
        if len(hyphen_joined) >= 3 and hyphen_joined not in PHARMA_STOPWORDS and not hyphen_joined.isdigit():
            tokens.add(hyphen_joined)
        # two word joins
        for i in range(len(words) - 1):
            joined = words[i] + words[i+1]
            if len(joined) >= 4 and joined not in PHARMA_STOPWORDS and not joined.isdigit():
                tokens.add(joined)
    return sorted(tokens)

for name, lines in sample_test_lines.items():
    print(f"\n=== Test: {name} ===")
    tokens = extract_candidate_tokens(lines)
    print("Candidate tokens:", tokens)
    
    brand_matches = {}
    for tok in tokens:
        if tok in brand_to_ids:
            brand_matches[tok] = max(brand_matches.get(tok, 0), 100.0)
        elif len(tok) >= 4:
            f_matches = process.extract(tok, brand_words_list, scorer=fuzz.ratio, limit=3, score_cutoff=75)
            for m_b, score, _ in f_matches:
                brand_matches[m_b] = max(brand_matches.get(m_b, 0), score)
    
    sorted_brands = sorted(brand_matches.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top matched brand words:", sorted_brands)
