import requests
import re
from pathlib import Path

BASE_URL = 'http://127.0.0.1:8000'
photos = [
    ('Brufen 30 tablets (180 deg)', 'tools/missed_samples/Brufen 30 tablets/huawei p30 (110).jpg'),
    ('Betadine 60 g ointment (180 deg)', 'tools/missed_samples/Betadine 60  g ointment/huawei p30 (397).jpg'),
    ('Fucidin 20 g cream (salt-first)', 'tools/missed_samples/Fucidin 20 g cream/huawei p30 (522).jpg'),
]

session = requests.Session()
r_init = session.get(f'{BASE_URL}/')
csrf_token = session.cookies.get('csrftoken')
if not csrf_token:
    m = re.search(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"', r_init.text)
    if m:
        csrf_token = m.group(1)

print(f"Obtained CSRF Token: {csrf_token[:16]}... from live server")

for label, p_str in photos:
    p = Path(p_str)
    print('\n' + '=' * 65)
    print('Testing live server:', label)
    print('File:', p)
    with open(p, 'rb') as f:
        files = {'image': (p.name, f, 'image/jpeg')}
        data = {'mode': 'medicine_package', 'csrfmiddlewaretoken': csrf_token}
        headers = {'X-CSRFToken': csrf_token, 'Referer': f'{BASE_URL}/'}
        resp = session.post(f'{BASE_URL}/scan/', files=files, data=data, headers=headers)
    print('HTTP Status:', resp.status_code)
    res_json = resp.json()
    print('Response status  :', res_json.get('status'))
    print('Rotation chosen  :', res_json.get('rotation'))
    print('Detected medicine:', res_json.get('detected_name'))
    top_cand = res_json.get('medicine') or (res_json.get('candidates') and res_json['candidates'][0]) or {}
    print('Top candidate    :', top_cand.get('brand'), f"({top_cand.get('name')})", '| score:', top_cand.get('score'))
    print('Candidates count :', len(res_json.get('candidates', [])))
    for idx, c in enumerate(res_json.get('candidates', [])[:3], 1):
        print(f"   {idx}. {c.get('brand')} ({c.get('name')}) — score: {c.get('score')} | salt_matched: {c.get('salt_matched')}")
