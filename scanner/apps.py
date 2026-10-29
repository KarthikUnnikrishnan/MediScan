import os
import sys
import threading
from django.apps import AppConfig


class ScannerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'scanner'

    def ready(self):
        # Guard against duplicate model loading on the 4 GB GPU:
        # Django's runserver starts twice when using the auto-reloader.
        # Only launch the background warm-up thread when RUN_MAIN == 'true' or --noreload is used.
        is_runserver = any('runserver' in arg for arg in sys.argv)
        is_reloader_child = os.environ.get('RUN_MAIN') == 'true'
        is_noreload = any('--noreload' in arg for arg in sys.argv)

        if is_runserver and (is_reloader_child or is_noreload):
            # FIX 0a: Print environment diagnostics once at server startup
            missing_pkgs = []
            versions = {}
            for pkg, pip_name in [('torch', 'torch'), ('transformers', 'transformers'), ('easyocr', 'easyocr'), ('rapidfuzz', 'rapidfuzz')]:
                try:
                    m = __import__(pkg)
                    versions[pkg] = getattr(m, '__version__', 'ok')
                except ImportError:
                    missing_pkgs.append(pip_name)

            cuda_str = "False"
            if 'torch' in versions:
                import torch
                if torch.cuda.is_available():
                    cuda_str = f"True ({torch.cuda.get_device_name(0)})"

            print(f"[MediScan] Python: {sys.executable}")
            print(
                f"[MediScan] torch={versions.get('torch', 'MISSING')} (CUDA={cuda_str}), "
                f"transformers={versions.get('transformers', 'MISSING')}, "
                f"easyocr={versions.get('easyocr', 'MISSING')}, "
                f"rapidfuzz={versions.get('rapidfuzz', 'MISSING')}"
            )

            if missing_pkgs:
                print(f"[MediScan ERROR] Missing required package(s): {', '.join(missing_pkgs)}")
                print(f'[MediScan FIX] Run: "{sys.executable}" -m pip install {" ".join(missing_pkgs)}')
            elif 'torch' in versions and cuda_str == 'False':
                print(
                    '[MediScan PERFORMANCE] CUDA is unavailable in this Python environment; '
                    'printed OCR will run on CPU and may be slow. Use a CUDA-enabled PyTorch '
                    'environment on machines with a supported NVIDIA GPU.'
                )

            def _warmup_all():
                try:
                    from .ml.medicine_index import get_brand_word_index
                    from .ml.recognisers import warmup_printed_recogniser
                    get_brand_word_index()
                    warmup_printed_recogniser()
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).error("Model warm-up error: %s", exc, exc_info=True)

            thread = threading.Thread(
                target=_warmup_all,
                name="MediScanWarmupThread",
                daemon=True,
            )
            thread.start()

