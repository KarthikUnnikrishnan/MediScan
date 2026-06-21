import albumentations as A
import cv2
import os
from pathlib import Path

# ── WHERE YOUR DATA LIVES ──────────────────────────────────────────────
SOURCE_IMAGES = r"D:\Coding Section\Mediscan\Datasets\Model1_Strip OCR\DS05_ultralytics_pills\train\images"
SOURCE_LABELS = r"D:\Coding Section\Mediscan\Datasets\Model1_Strip OCR\DS05_ultralytics_pills\train\labels"

# ── WHERE AUGMENTED DATA WILL BE SAVED ────────────────────────────────
OUTPUT_IMAGES = r"D:\Coding Section\Mediscan\training\augmented_yolo\images"
OUTPUT_LABELS = r"D:\Coding Section\Mediscan\training\augmented_yolo\labels"

os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)

# ── AUGMENTATION PIPELINE ─────────────────────────────────────────────
# Each transform randomly changes the image in a different way.
# bbox_params tells Albumentations to also adjust bounding boxes automatically.
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.8),
    A.GaussianBlur(blur_limit=(3, 7), p=0.4),
    A.ImageCompression(quality_lower=55, quality_upper=95, p=0.5),
    A.Perspective(scale=(0.04, 0.10), p=0.4),
    A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=30, val_shift_limit=25, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.RandomShadow(p=0.2),
], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    min_visibility=0.30  # discard a box if less than 30% of it remains after crop
))

VARIANTS_PER_IMAGE = 12  # 92 images × 12 = 1,104 augmented + 92 originals = 1,196 total


# ── HELPER: read a YOLO .txt label file ───────────────────────────────
def read_labels(label_path):
    boxes, classes = [], []
    if not os.path.exists(label_path):
        return boxes, classes
    with open(label_path, 'r') as f:
        for line in f.read().strip().splitlines():
            if line.strip():
                parts = line.strip().split()
                classes.append(int(parts[0]))
                boxes.append([float(x) for x in parts[1:5]])
    return boxes, classes


# ── HELPER: write a YOLO .txt label file ──────────────────────────────
def write_labels(label_path, boxes, classes):
    with open(label_path, 'w') as f:
        for cls, box in zip(classes, boxes):
            f.write(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")


# ── MAIN LOOP ─────────────────────────────────────────────────────────
image_files = [
    f for f in os.listdir(SOURCE_IMAGES)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

print(f"Found {len(image_files)} source images. Generating {VARIANTS_PER_IMAGE} variants each...")
total_saved = 0

for img_file in image_files:
    stem = Path(img_file).stem
    image_path = os.path.join(SOURCE_IMAGES, img_file)
    label_path = os.path.join(SOURCE_LABELS, stem + '.txt')

    # Read image (OpenCV loads as BGR, Albumentations needs RGB)
    image = cv2.imread(image_path)
    if image is None:
        print(f"  Skipping unreadable image: {img_file}")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes, classes = read_labels(label_path)

    # Save the original image as-is into the output folder
    cv2.imwrite(
        os.path.join(OUTPUT_IMAGES, img_file),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )
    write_labels(os.path.join(OUTPUT_LABELS, stem + '.txt'), boxes, classes)
    total_saved += 1

    # Generate augmented variants
    for i in range(VARIANTS_PER_IMAGE):
        try:
            result = transform(image=image, bboxes=boxes, class_labels=classes)
            aug_img   = result['image']
            aug_boxes = list(result['bboxes'])
            aug_cls   = result['class_labels']

            out_stem = f"{stem}_aug{i:03d}"
            cv2.imwrite(
                os.path.join(OUTPUT_IMAGES, out_stem + '.jpg'),
                cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            )
            write_labels(
                os.path.join(OUTPUT_LABELS, out_stem + '.txt'),
                aug_boxes, aug_cls
            )
            total_saved += 1

        except Exception as e:
            print(f"  Warning on {img_file} variant {i}: {e}")

print(f"\nDone! Saved {total_saved} images to: training/augmented_yolo/")
print(f"Original: {len(image_files)}  |  After augmentation: {total_saved}")