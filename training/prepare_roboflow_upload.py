import os
import shutil
from pathlib import Path

# Source: the folder with 151 brand subfolders
SOURCE = r"D:\Coding Section\Mediscan\Datasets\Model1_Strip OCR\Mobile-Captured Pharmaceutical Medication Packages"

# Destination: a flat folder with all images ready to upload
DEST = r"D:\Coding Section\Mediscan\training\roboflow_upload"

os.makedirs(DEST, exist_ok=True)

# How many images to collect per brand folder
# 151 folders × 3 images = ~450 images total
IMAGES_PER_FOLDER = 3

collected = 0
skipped = 0

for brand_folder in sorted(os.listdir(SOURCE)):
    brand_path = os.path.join(SOURCE, brand_folder)
    if not os.path.isdir(brand_path):
        continue

    # Get all images in this brand folder
    images = [
        f for f in os.listdir(brand_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not images:
        skipped += 1
        continue

    # Take up to IMAGES_PER_FOLDER images from each brand
    selected = images[:IMAGES_PER_FOLDER]

    for img_file in selected:
        src_path = os.path.join(brand_path, img_file)

        # Rename to include brand name so filenames stay unique
        safe_brand = brand_folder.replace(' ', '_').replace('/', '-')
        new_name = f"{safe_brand}__{img_file}"
        dest_path = os.path.join(DEST, new_name)

        shutil.copy2(src_path, dest_path)
        collected += 1

print(f"\nDone!")
print(f"Collected : {collected} images")
print(f"Skipped   : {skipped} empty folders")
print(f"Saved to  : {DEST}")
print(f"\nNow upload the folder '{DEST}' to Roboflow.")