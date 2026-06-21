import os
import shutil
from ultralytics import YOLO
import torch

# ── CHECK GPU ─────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = 0  # GPU
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU detected: {gpu_name}")
    print(f"Training will use GPU — estimated time: 1-2 hours\n")
else:
    device = 'cpu'
    print("No GPU detected — training on CPU")
    print("Estimated time on CPU: 6-12 hours")
    print("TIP: Let it run overnight, or use Google Colab (free GPU)\n")

# ── PATHS ─────────────────────────────────────────────────────────────
DATA_YAML    = r"D:\Coding Section\Mediscan\training\final_dataset\data.yaml"
OUTPUT_DIR   = r"D:\Coding Section\Mediscan\training\yolo_runs"
SAVE_DIR     = r"D:\Coding Section\Mediscan\saved_models"

os.makedirs(SAVE_DIR, exist_ok=True)

# ── LOAD PRE-TRAINED MODEL ────────────────────────────────────────────
# yolov8s.pt = YOLOv8 Small — pre-trained on COCO (80 classes, 118K images)
# We're fine-tuning it for 1 class: medicine_strip
print("Loading pre-trained YOLOv8s weights...")
model = YOLO('yolov8s.pt')  # Downloads automatically on first run (~22MB)

# ── TRAIN ─────────────────────────────────────────────────────────────
print("Starting fine-tuning...\n")
results = model.train(
    data     = DATA_YAML,
    epochs   = 60,          # 60 passes through all training images
    imgsz    = 640,          # resize all images to 640×640 during training
    batch    = 8,            # 8 images processed together (lower if RAM error)
    device   = device,
    patience = 15,           # stop early if no improvement for 15 epochs
    save     = True,
    project  = OUTPUT_DIR,
    name     = 'strip_detector',
    exist_ok = True,
    workers  = 0,

    # Augmentation during training (on top of your pre-augmented data)
    hsv_h    = 0.015,
    hsv_s    = 0.5,
    hsv_v    = 0.4,
    flipud   = 0.1,
    fliplr   = 0.5,
    mosaic   = 0.8,
)

# ── SAVE BEST WEIGHTS ─────────────────────────────────────────────────
best_weights = os.path.join(OUTPUT_DIR, 'strip_detector', 'weights', 'best.pt')
save_path    = os.path.join(SAVE_DIR, 'strip_detector.pt')

if os.path.exists(best_weights):
    shutil.copy2(best_weights, save_path)
    print(f"\nBest model saved to: {save_path}")
else:
    print("\nWarning: best.pt not found — check yolo_runs/strip_detector/weights/")

# ── PRINT RESULTS ─────────────────────────────────────────────────────
print("\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)
print(f"mAP@50  : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.3f}")
print(f"mAP@50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.3f}")
print(f"\nWeights saved: {save_path}")
print("Next step: test with a real medicine photo")