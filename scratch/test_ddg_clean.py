import sys
sys.path.insert(0, '.')
import requests
import re
from scanner.ml.price_verifier import extract_package_details, extract_prices_from_text

s = requests.Session()
s.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

r = s.post('https://html.duckduckgo.com/html/', data={'q': 'Levoz 5mg Tablet price 1mg'}, timeout=6)
print('Status:', r.status_code)
snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', r.text, re.DOTALL)
for i, sn in enumerate(snippets[:4]):
    clean = re.sub(r'<[^>]+>', '', sn).strip()
    print(f'[{i}] {clean}')
    qty, unit, basis = extract_package_details(clean)
    prices = extract_prices_from_text(clean)
    print('   -> qty:', qty, unit, 'prices:', prices)
