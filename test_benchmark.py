import os
import time
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediscan.settings')
django.setup()

from scanner.ml import scan_image

test_dir = 'test_images'
for fname in sorted(os.listdir(test_dir)):
    fpath = os.path.join(test_dir, fname)
    t0 = time.time()
    try:
        res = scan_image(fpath, mode='auto')
        dur = time.time() - t0
        med_name = res.get('medicine', {}).get('name') if res.get('medicine') else None
        if not med_name and res.get('medicines'):
            med_name = [m.get('name') for m in res.get('medicines')]
        print(f"{fname:30} | {dur:5.2f}s | {res.get('status'):10} | {res.get('mode'):16} | {str(med_name)[:45]}")
    except Exception as e:
        print(f"{fname:30} | ERROR: {e}")
