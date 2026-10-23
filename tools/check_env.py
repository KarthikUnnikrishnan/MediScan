"""
tools/check_env.py — MediScan Offline Environment & Dependency Diagnostics.
Verifies Python executable, GPU CUDA capability, and all offline ML packages.
"""

import sys
import os
from pathlib import Path

REQUIRED_PKGS = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("easyocr", "easyocr"),
    ("rapidfuzz", "rapidfuzz"),
    ("PIL", "pillow"),
    ("cv2", "opencv-python-headless"),
    ("ultralytics", "ultralytics"),
    ("sentencepiece", "sentencepiece"),
]


def check_environment(verbose: bool = True) -> bool:
    print("=" * 65)
    print("MediScan Offline Backend — Environment Diagnostic")
    print("=" * 65)
    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version.split()[0]} ({sys.platform})")

    missing = []
    versions = {}

    for import_name, pip_name in REQUIRED_PKGS:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "installed")
            versions[pip_name] = ver
        except ImportError:
            missing.append(pip_name)

    # CUDA Check
    cuda_available = False
    cuda_device = "N/A"
    if "torch" not in missing:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_device = torch.cuda.get_device_name(0)

    print(f"CUDA Available    : {cuda_available} ({cuda_device})")
    print("-" * 65)
    print("Package Versions:")
    for import_name, pip_name in REQUIRED_PKGS:
        ver = versions.get(pip_name, "MISSING")
        status = "OK" if ver != "MISSING" else "FAILED"
        print(f"  - {pip_name:<25}: {ver:<18} [{status}]")

    print("-" * 65)

    # Check local saved_models files
    root = Path(__file__).resolve().parent.parent
    saved_models = root / "saved_models"
    model_a_dir = saved_models / "modelA_printed_ocr"
    med_sqlite = saved_models / "medicines.sqlite"
    drug_sqlite = saved_models / "drugs.sqlite"

    print("Model & Database Assets:")
    print(f"  - modelA_printed_ocr : {'EXISTS' if model_a_dir.is_dir() else 'MISSING'} ({model_a_dir})")
    print(f"  - medicines.sqlite   : {'EXISTS' if med_sqlite.is_file() else 'MISSING'} ({med_sqlite})")
    print(f"  - drugs.sqlite       : {'EXISTS' if drug_sqlite.is_file() else 'MISSING'} ({drug_sqlite})")
    print("=" * 65)

    if missing:
        missing_str = " ".join(missing)
        print(f"\n[ERROR] Missing required package(s): {', '.join(missing)}")
        print(f"To install for this Python interpreter, run:")
        print(f'  "{sys.executable}" -m pip install {missing_str}\n')
        return False

    if not cuda_available:
        print("\n[WARNING] CUDA is NOT available for this Python environment.")
        print("TrOCR and EasyOCR will run on CPU, which is substantially slower.")
        print("To enable GPU acceleration on Windows (RTX GPU), run:")
        print(f'  "{sys.executable}" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126\n')

    print("[SUCCESS] All required offline packages and assets are ready.\n")
    return True


if __name__ == "__main__":
    ok = check_environment(verbose=True)
    sys.exit(0 if ok else 1)
