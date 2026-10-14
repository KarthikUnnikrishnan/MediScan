import os
import django
import time

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediscan.settings')
django.setup()

from PIL import Image
from google import genai
from google.genai import types
from scanner.ml.vision import load_models, run_ocr, crop_strip
from scanner.ml.gemini_vision import _SYSTEM_PROMPT, _extract_response_text

def compute_cer(ref: str, hyp: str) -> float:
    r = ref.lower().strip()
    h = hyp.lower().strip()
    if not r:
        return 1.0 if h else 0.0
    m, n = len(r), len(h)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if r[i - 1] == h[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n] / max(len(r), 1)

ground_truth = {
    'test_images/strip.jpg': {
        'brand': '1-AL',
        'ingredient': 'levocetirizine',
        'strength': '5 mg',
        'ref_text': '1-AL Levocetirizine Tablets IP 5 mg FDC Limited'
    },
    'test_images/LORI0006_L.jpg': {
        'brand': 'Lorinol',
        'ingredient': 'loratadine',
        'strength': '10 mg',
        'ref_text': 'Lorinol 10 Loratadine Tablets IP 10 mg MICRO LABS LIMITED'
    },
    'test_images/paracetamol.jpg': {
        'brand': 'P-650',
        'ingredient': 'paracetamol',
        'strength': '650 mg',
        'ref_text': 'P-650 Paracetamol Tablets IP 650 mg apex laboratories'
    }
}

load_models()
api_key = os.environ.get('GEMINI_API_KEY')
client = genai.Client(api_key=api_key)

models = ['TrOCR', 'gemini-3.1-flash-lite', 'gemini-3.5-flash-lite']
results = {m: {'samples': 0, 'success': 0, 'failed': 0, 'cer_list': [], 'brand_correct': 0, 'ing_correct': 0, 'str_correct': 0} for m in models}

for img_path, gt in ground_truth.items():
    img = Image.open(img_path)
    crop, _ = crop_strip(img)

    # 1. TrOCR
    m = 'TrOCR'
    results[m]['samples'] += 1
    t_text = run_ocr(crop)
    if t_text:
        results[m]['success'] += 1
        cer = compute_cer(gt['ref_text'], t_text)
        results[m]['cer_list'].append(cer)
        if gt['brand'].lower() in t_text.lower():
            results[m]['brand_correct'] += 1
        if gt['ingredient'].lower() in t_text.lower():
            results[m]['ing_correct'] += 1
        if gt['strength'].lower() in t_text.lower():
            results[m]['str_correct'] += 1
    else:
        results[m]['failed'] += 1
        results[m]['cer_list'].append(1.0)
    print(f"[TrOCR] {img_path}: {t_text!r} | CER: {results[m]['cer_list'][-1]:.3f}")

    # Gemini models
    with open(img_path, 'rb') as f:
        img_bytes = f.read()

    for g_mod in ['gemini-3.1-flash-lite', 'gemini-3.5-flash-lite']:
        results[g_mod]['samples'] += 1
        try:
            resp = client.models.generate_content(
                model=g_mod,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'),
                    _SYSTEM_PROMPT
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    http_options=types.HttpOptions(timeout=10000, retry_options=types.HttpRetryOptions(attempts=1))
                )
            )
            raw = _extract_response_text(resp)
            results[g_mod]['success'] += 1
            cer = compute_cer(gt['ref_text'], raw[:len(gt['ref_text']) * 2])
            results[g_mod]['cer_list'].append(cer)
            if gt['brand'].lower() in raw.lower():
                results[g_mod]['brand_correct'] += 1
            if gt['ingredient'].lower() in raw.lower():
                results[g_mod]['ing_correct'] += 1
            if gt['strength'].lower() in raw.lower():
                results[g_mod]['str_correct'] += 1
            print(f"[{g_mod}] {img_path}: Success | CER: {cer:.3f}")
        except Exception as exc:
            results[g_mod]['failed'] += 1
            results[g_mod]['cer_list'].append(1.0)
            err_type = type(exc).__name__
            print(f"[{g_mod}] {img_path}: Failed ({err_type}: {str(exc)[:60]})")

print('\n=== BENCHMARK SUMMARY ===')
for m, r in results.items():
    avg_cer = sum(r['cer_list']) / len(r['cer_list']) if r['cer_list'] else 1.0
    print(f"Model: {m}")
    print(f"  Samples: {r['samples']}, Succeeded: {r['success']}, Failed: {r['failed']}")
    print(f"  Average CER: {avg_cer:.3f}")
    print(f"  Brand Accuracy: {r['brand_correct']}/{r['samples']} ({r['brand_correct']/r['samples']*100:.1f}%)")
    print(f"  Ingredient Accuracy: {r['ing_correct']}/{r['samples']} ({r['ing_correct']/r['samples']*100:.1f}%)")
    print(f"  Strength Accuracy: {r['str_correct']}/{r['samples']} ({r['str_correct']/r['samples']*100:.1f}%)")
