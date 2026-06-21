from django.apps import AppConfig

class MlConfig(AppConfig):
    name = 'ml'

    def ready(self):
        # Load all ML models once when Django starts
        # This prevents reloading on every request
        import importlib, sys

        loaders = [
            ('ml.strip_ocr',       'load_models', 'load_strip'),
            ('ml.prescription_htr','load_models', 'load_htr'),
            ('ml.generic_finder',  'load_db',     'load_generic'),
            ('ml.drug_info',       'load_db',     'load_drug'),
        ]

        print("\n[MediScan] Loading all ML models...")
        for module_path, func_name, alias in loaders:
            try:
                mod = importlib.import_module(module_path)
                getattr(mod, func_name)()
            except ModuleNotFoundError as e:
                print(f"[MediScan] WARNING: Could not load {module_path}: {e}")
            except Exception as e:
                print(f"[MediScan] WARNING: Error loading {module_path}: {e}")
        print("[MediScan] ML model loading complete.\n")