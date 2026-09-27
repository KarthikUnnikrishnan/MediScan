import os
import sys
import django
from PIL import Image

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediscan.settings')
django.setup()

from scanner.ml.vision import load_models, crop_strip, run_ocr
from scanner.ml import scan_image

def main():
    print("=" * 60)
    print("Initializing vision models via load_models()...")
    load_models()
    print("Models initialized successfully!")
    print("=" * 60)

    from django.conf import settings
    images = {
        "strip.jpg": str(settings.BASE_DIR / "test_images" / "strip.jpg"),
        "prescription.jpg": str(settings.BASE_DIR / "test_images" / "prescription.jpg"),
    }

    for name, img_path in images.items():
        print(f"\nTesting: {name} ({img_path})")
        print("-" * 50)
        img = Image.open(img_path)
        print(f"Original image size: {img.size}")

        # Direct test of crop_strip
        cropped, yolo_hit = crop_strip(img)
        print(f"crop_strip() result: YOLO hit={yolo_hit}, Cropped size={cropped.size}")

        # Direct test of run_ocr
        ocr_text = run_ocr(cropped)
        print(f"run_ocr(cropped) result: '{ocr_text}'")

        # Full pipeline test via scan_image
        print("Running full pipeline scan_image()...")
        scan_result = scan_image(img_path)
        print(f"scan_image status: {scan_result.get('status')}")
        print(f"scan_image success: {scan_result.get('success')}")
        print(f"scan_image raw_ocr: {scan_result.get('raw_ocr')}")
        if 'medicine' in scan_result and scan_result['medicine']:
            med = scan_result['medicine']
            print(f"Matched Medicine: {med.get('name')} | Manufacturer: {med.get('manufacturer')} | Price: {med.get('price')}")
        if 'medicines' in scan_result and scan_result['medicines']:
            print(f"Matched Medicines ({len(scan_result['medicines'])}):")
            for m in scan_result['medicines']:
                print(f"  - {m.get('name')} | {m.get('salt_composition')}")
        if 'alternatives' in scan_result and scan_result['alternatives']:
            print(f"Found {len(scan_result['alternatives'])} alternatives:")
            for alt in scan_result['alternatives'][:3]:
                print(f"  - {alt.get('name')} (Price: {alt.get('price')}, Savings: {alt.get('savings_percent')}%)")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)

if __name__ == '__main__':
    main()
