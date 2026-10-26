"""Live Django upload regressions for the medicine package photos."""
from pathlib import Path
import os
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mediscan.settings")

import django
django.setup()

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client

PHOTOS = ROOT / "tools" / "test_photos"
CASES = [
    ("Lorinol-10", "LORI0006_L.jpg", ["Lorinol 10 Tablet", "alternatives"]),
    ("Fexomine-180", "images.jpg", ["Fexomine", "strength", "Likely"]),
    ("Levocetirizine FDC", "strip.jpg", ["Levocetirizine", "5 mg", "FDC"]),
    ("P-650", "paracetamol.jpg", ["P-650", "Paracetamol", "650 mg", "Apex"]),
]

client = Client()
failures = []
for case, filename, required in CASES:
    path = PHOTOS / filename
    with path.open("rb") as image:
        upload = SimpleUploadedFile(path.name, image.read(), content_type="image/jpeg")
    response = client.post("/", {"image": upload, "mode": "medicine_package"}, follow=True)
    html = response.content.decode("utf-8", errors="replace")
    print(f"\n[{case}] HTTP {response.status_code} final_url={response.request.get('PATH_INFO')}")
    result = client.session.get("scan_result", {})
    med = result.get("medicine") or {}
    print(f"  decision: level={result.get('result_level')} medicine={med.get('name')} salt={med.get('salt')} manufacturer={med.get('manufacturer')}")
    print(f"  message: {result.get('message')}")
    checks = [(f"contains {text!r}", text.casefold() in html.casefold()) for text in required]
    if case == "Lorinol-10":
        checks.append(("rendered level is Verified", re.search(r'class="badge-verified">\s*Verified match', html) is not None))
    if case == "Fexomine-180":
        checks += [("not marked Verified", re.search(r'class="badge-verified">\s*Verified match', html) is None), ("has mismatch wording", "strength mismatch" in html.casefold())]
    if case == "Levocetirizine FDC":
        checks += [("excludes Sted", re.search(r"\bsted\b", html, re.I) is None), ("excludes Telio", re.search(r"\btelio\b", html, re.I) is None)]
    if case == "P-650":
        checks += [(f"excludes {word}", word.casefold() not in html.casefold()) for word in ("Cetirizine", "Phenylephrine", "Dava")]
    for label, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'} {label}")
        if not passed:
            failures.append(f"{case}: {label}")

print(f"\nSummary: {len(failures)} failed assertion(s)")
for failure in failures:
    print(f"  - {failure}")
if failures:
    raise SystemExit(1)
