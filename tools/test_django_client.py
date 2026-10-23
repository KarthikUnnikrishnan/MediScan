"""
tools/test_django_client.py — Test MediScan offline scan pipeline via Django test client.

Posts 3 photos to /scan/ in mode 'medicine_package', prints JSON results,
and posts to /scan/ in mode 'prescription' to verify the 'not installed yet' message reaches the response.
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediscan.settings')
import django
django.setup()

from django.test import Client
from django.core.files.uploadedfile import SimpleUploadedFile


def main():
    print("=" * 80)
    print("TESTING SCAN VIEWS VIA DJANGO TEST CLIENT")
    print("=" * 80)

    client = Client()

    # 3 sample test photos
    test_photos = [
        Path(r"D:\MediScan_data\ModelA\raw\mobile_pharma_packages\bjy2svvmn8-1\Acretin 30 g cream\huawei cn (48).jpg"),
        Path(r"D:\MediScan_data\ModelA\raw\mobile_pharma_packages\bjy2svvmn8-1\Adol 24 caplets\huawei cn (131).jpg"),
        Path(r"D:\MediScan_data\ModelA\raw\mobile_pharma_packages\bjy2svvmn8-1\Aggrex 60 tablets\huawei cn (66).jpg"),
    ]

    # TEST 1: 3 photos in mode medicine_package
    print("\n--- TEST 1: Post 3 photos in mode 'medicine_package' ---")
    for idx, photo_path in enumerate(test_photos, 1):
        print(f"\n[1.{idx}] Testing: {photo_path.name} ({photo_path.parent.name})")
        if not photo_path.exists():
            print(f"Warning: {photo_path} does not exist!")
            continue

        with open(photo_path, "rb") as f:
            uploaded_file = SimpleUploadedFile(photo_path.name, f.read(), content_type="image/jpeg")

        response = client.post(
            "/scan/",
            {"image": uploaded_file, "mode": "medicine_package"},
        )

        print(f"HTTP Status: {response.status_code}")
        try:
            data = response.json()
            print(f"Success    : {data.get('success')}")
            print(f"Status     : {data.get('status')}")
            print(f"Detected   : {data.get('detected_name') or (data.get('medicine') or {}).get('name')}")
            print(f"Candidates : {len(data.get('candidates', []))} candidates returned")
            for c_idx, cand in enumerate(data.get('candidates', []), 1):
                print(f"   Cand {c_idx}: {cand.get('brand')} | score={cand.get('score')} | confident={cand.get('confident')} | corroborated={cand.get('corroborated')}")
            if data.get('alternatives'):
                print(f"Alternatives: {len(data['alternatives'])} cheaper generics found (Max saving: Rs {data.get('max_savings')})")
            if data.get('side_effects'):
                print(f"Side effects: {data['side_effects'][:3]}")
        except Exception as e:
            print("Failed to decode JSON:", e, response.content[:200])

    # TEST 2: mode prescription
    print("\n" + "-" * 80)
    print("--- TEST 2: Post photo in mode 'prescription' ---")
    sample_rx = test_photos[0]
    with open(sample_rx, "rb") as f:
        uploaded_rx = SimpleUploadedFile(sample_rx.name, f.read(), content_type="image/jpeg")

    rx_response = client.post(
        "/scan/",
        {"image": uploaded_rx, "mode": "prescription"},
    )

    print(f"HTTP Status: {rx_response.status_code}")
    rx_data = rx_response.json()
    print("Response JSON:")
    print(json.dumps(rx_data, indent=2))

    assert "not installed yet" in rx_data.get("message", "").lower(), "Expected 'not installed yet' in message!"
    print("\nPASS: 'Handwritten model is not installed yet' message confirmed in client response.")


if __name__ == "__main__":
    main()
