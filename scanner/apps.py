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
            def _warmup_all():
                from .ml.medicine_index import get_brand_word_index
                from .ml.recognisers import warmup_printed_recogniser
                get_brand_word_index()
                warmup_printed_recogniser()

            thread = threading.Thread(
                target=_warmup_all,
                name="MediScanWarmupThread",
                daemon=True,
            )
            thread.start()
