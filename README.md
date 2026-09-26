# MediScan 🌿 — AI-Powered Medicine Strip & Prescription Scanner

MediScan is a full-stack Django application that leverages computer vision (YOLOv8) and deep learning OCR (TrOCR) to scan medicine packaging and handwritten or printed prescriptions. It extracts active salts, matches brand and generic medicines against comprehensive databases, computes cost-saving generic alternatives, and performs automated drug-drug cross-interaction safety analysis.

---

## 🚀 Key Features

1. **Dual Scanning Modes**:
   - **Mode 1 (Medicine Package / Strip)**: Detects blister pack/box boundaries using a custom YOLOv8 detector, crops the label, runs TrOCR text recognition, and matches active pharmaceutical ingredients.
   - **Mode 2 (Prescription / Multi-medicine)**: Performs full-document OCR, extracts multiple medicine names, looks up their clinical profiles, and conducts automated cross-interaction conflict checking.
2. **Generic Alternatives & Price Comparison**: Identifies matching formulations and calculates cost-saving alternative brands (often saving 40%–80%).
3. **Safety & Side Effects Database**: Fetches MedDRA-standardized side effects, occurrence frequencies, and high-risk drug interaction warnings.
4. **Modern UI/UX**: Responsive glassmorphism interface featuring Google Fonts (`Fraunces` & `Plus Jakarta Sans`), live camera capture, drag-and-drop file upload, interactive results view, and client-side scan history drawer.

---

## 📁 Repository Structure

```text
MediScan/
├── manage.py                     # Django management CLI script
├── requirements.txt              # Python project dependencies
├── .env                          # Local environment configuration
├── .gitignore                    # Git exclude rules (weights, venv, pycache)
│
├── mediscan/                     # Root Django project configuration
│   ├── __init__.py
│   ├── settings.py               # Django settings, app configs, media/static paths
│   ├── urls.py                   # Master URL dispatching
│   └── wsgi.py                   # WSGI application entry point
│
├── scanner/                      # Core scanner application
│   ├── apps.py                   # App configuration
│   ├── forms.py                  # ScanForm validation (image files, scan mode)
│   ├── models.py                 # (Optional persistent records)
│   ├── urls.py                   # Routes: index, scan AJAX endpoint, result redirect
│   ├── views.py                  # HTTP request handlers & session storage
│   ├── ml.py                     # ML pipeline (YOLOv8 + TrOCR + SQLite matchers)
│   └── templatetags/             # Custom Django template filters
│       └── mediscan_tags.py      # Math & formatting template tags
│
├── templates/                    # HTML UI templates
│   ├── base.html                 # Main layout, nav, camera modal, history drawer, footer
│   └── scanner/
│       ├── index.html            # Hero, scan card, sample tests, bento features, FAQ
│       └── result.html           # Scan analysis report view
│
├── static/                       # Custom static assets
│   ├── css/
│   │   └── main.css              # Master styling design system (glassmorphic, responsive)
│   └── js/
│       └── main.js               # Client interactivity (camera, drag & drop, history)
│
├── Datasets/                     # Dataset discovery & download documentation
│   └── Model1_Strip_OCR_Roboflow/
│       ├── discovery_report.txt
│       └── download_report.txt
│
└── saved_models/                 # Model checkpoints & configuration
    ├── strip_detector.pt         # (Excluded from code zip due to binary size)
    ├── drugs.sqlite              # (Excluded from code zip due to binary size)
    ├── medicines.sqlite          # (Excluded from code zip due to binary size)
    └── strip_ocr_finetuned/      # Fine-tuned TrOCR model configuration & tokenizer
        ├── config.json
        ├── generation_config.json
        ├── tokenizer_config.json
        ├── preprocessor_config.json
        ├── special_tokens_map.json
        └── eval_report.txt
```

---

## 🧠 Machine Learning & Data Pipeline (`scanner/ml.py`)

1. **Object Detection**:
   - Model: Custom fine-tuned YOLOv8 (`strip_detector.pt`)
   - Function: Locates medicine strips, blister packs, and bottles in cluttered backgrounds.
2. **Text Recognition (OCR)**:
   - Model: Fine-tuned `microsoft/trocr-base-stage1` (`strip_ocr_finetuned/`)
   - Preprocessing: Grayscale / contrast enhancement, image resizing to 384×384.
3. **Fuzzy String Matching & Entity Extraction**:
   - Compares recognized text against drug names and salt formulations using `difflib.get_close_matches` and regex normalization.
4. **Knowledge Base / SQLite Schema**:
   - `medicines.sqlite`: Table `medicines(id, name, salt, manufacturer, price, source)`
   - `drugs.sqlite`: Tables `drug_interactions(drug1, drug2, description)`, `side_effects(...)`, `se_frequency(...)`, `drug_stitch_map(...)`.

---

## 🛠️ Installation & Setup

1. **Clone repository & enter directory**:
   ```bash
   cd MediScan
   ```

2. **Set up Python Virtual Environment**:
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Configure `.env`:
   ```env
   SECRET_KEY=your-secure-secret-key
   DEBUG=True
   ```

5. **Run Migrations & Start Server**:
   ```bash
   python manage.py migrate
   python manage.py runserver
   ```
   Access the application at `http://127.0.0.1:8000`.
