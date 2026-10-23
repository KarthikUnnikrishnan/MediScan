# MediScan 🌿 — System Architecture, Overall Working & Future Roadmap

> **Document Purpose**: This comprehensive technical guide details the full-stack architecture, machine learning pipeline, data schemas, verification heuristics, and execution workflow of **MediScan**. It is designed to provide complete context for architectural reviews, engineering discussions, and planning future implementations with AI systems (such as ChatGPT) and developer teams.

---

## 1. Executive Summary & Clinical Value Proposition

**MediScan** is an offline, privacy-first healthcare intelligence platform built with **Django**, **PyTorch / Hugging Face Transformers (TrOCR)**, **Ultralytics YOLOv8**, and **SQLite**. 

In many healthcare ecosystems (particularly across India and developing nations):
1. **Prescription illegibility and packaging confusion** cause medication errors.
2. **Branded medicine markups** place heavy financial burdens on patients when equivalent bioequivalent generic formulations exist at **40% to 80% lower costs**.
3. **Polypharmacy (taking multiple medications simultaneously)** frequently leads to dangerous, unflagged adverse drug-drug interactions.

MediScan solves these problems by allowing users to photograph a **blister strip / medicine box** (Mode 1) or a **multi-medicine prescription sheet** (Mode 2), extracting the active chemical formulations, identifying verified products across a catalog of **503,000+ medicines**, calculating cheaper bioequivalent generic alternatives, and screening for **MedDRA side effects and cross-drug interactions** — all running **100% locally and offline without external API dependencies**.

---

## 2. High-Level Architecture & End-to-End Workflow

```
[ User Input ]
  ├── Live Camera Capture (WebRTC / Canvas)
  └── Image File Upload (Drag-and-Drop / File Picker)
        │
        ▼
[ Django Web Application: scanner/views.py ]
  ├── Form Validation (ScanForm: image format, max 10MB)
  ├── Session Management & Temp Storage
  └── Dispatches to ML Pipeline (scanner/ml/__init__.py: scan_image())
        │
        ▼
[ Core Vision & Detection: scanner/ml/vision.py ]
  ├── YOLOv8 Nano Detector (saved_models/strip_detector.pt)
  │     └── Bounding box inference -> crops blister pack / label region
  └── Fine-Tuned TrOCR (saved_models/strip_ocr_finetuned/)
        ├── Preprocessing: RGB conversion, contrast enhancement, 384x384 resize
        ├── Multi-pass text recognition (Crop pass + Full image pass + Enhanced pass)
        └── Anti-Hallucination Guard (_ocr_output_looks_real heuristic)
        │
        ▼
[ Clinical NLP & Formulation Extraction: scanner/ml/extractor.py & salt_parser.py ]
  ├── Text Normalization (strip noise, bracket cleaning, unit standardization)
  ├── Salt Parser (regex & dictionary tokenizer: extracts salt, strength, unit)
  ├── Dosage Form Extraction (Tablet, Capsule, Syrup, Injection, Drops, Gel, Ointment)
  └── Candidate Brand & Manufacturer Identification
        │
        ▼
[ Database Matching Engine: scanner/ml/medicine_matcher.py & medicine_index.py ]
  ├── Searches Dynamic 503k+ In-Memory Index (saved_models/medicine_index.pkl)
  ├── Multi-Tier Verification (Exact Brand -> Alias Match -> Salt + Strength Exact Match)
  ├── Strict Composition Hard-Gate (disallows matching if active salts or strengths conflict)
  └── Confidence Breakdown Score (Brand %, Ingredients %, Strength %, Manufacturer %, Form %)
        │
        ▼
[ Generic Alternatives & Price Verification: scanner/ml/alternative_matcher.py & price_verifier.py ]
  ├── Queries medicines.sqlite for identical salt(s) + identical strength(s) + matching form
  ├── Unit Price Normalization (converts strip price to per-tablet / per-ml unit price)
  ├── Calculates Max Savings (Rs. and % discount)
  └── Trust Scoring (WHO-GMP licensed manufacturer tiers & price transparency)
        │
        ▼
[ Drug Safety & Clinical Conflict Checking: scanner/ml/drug_safety.py ]
  ├── Queries drugs.sqlite for MedDRA-standardized Side Effects & Frequencies
  └── Cross-Drug Interaction Engine (pairwise conflict matrix between all prescribed drugs)
        │
        ▼
[ Presentation & Report: templates/scanner/result.html & static/js/main.js ]
  ├── Glassmorphic UI with dynamic tabs for multi-medicine prescriptions
  ├── Interactive savings cards & alternatives drawer
  └── Scan History Drawer (localStorage client persistence)
```

---

## 3. Modular ML & Backend Subsystems

The backend ML logic is structured cleanly within `scanner/ml/`:

### 3.1. Vision & Anti-Hallucination (`vision.py`)
- **YOLOv8 Detector (`strip_detector.pt`)**: Fine-tuned YOLOv8n object detection model trained to locate blister packs, strip edges, and medicine containers within complex, cluttered backgrounds.
- **TrOCR (`strip_ocr_finetuned/`)**: Fine-tuned `microsoft/trocr-base-stage1` vision encoder-decoder model. Converts pixels directly to BPE token IDs.
- **Anti-Hallucination Filter (`_ocr_output_looks_real`)**: Deep learning seq2seq OCR models are prone to "hallucinating" common training phrases when shown noisy foil or empty crops. MediScan guards against this by checking character entropy, repetition, and alphanumeric density. If text is unconvincing, it returns `""` and safely triggers unverified fallback rather than outputting false clinical data.

### 3.2. Salt Formulation Parser (`salt_parser.py`)
- Standardizes messy pharmaceutical strings into structured components:
  `[{"ingredient": "paracetamol", "canonical_name": "paracetamol", "strength": 650.0, "unit": "mg"}]`
- Resolves synonym salts (e.g., *acetaminophen* -> *paracetamol*, *cholecalciferol* -> *vitamin d3*).
- Handles multi-salt combinations with varying delimiters (`+`, `/`, `&`, `with`).

### 3.3. Context-Aware Structured Extractor (`extractor.py`)
- Distinguishes between **Brand Names**, **Generic Compositions**, **Dosage Forms**, **Strengths**, and **Manufacturer Names** from raw OCR text.
- Reconstructs descriptive clinical titles (e.g., *"Calpol 500 Tablet (Paracetamol 500mg)"*).

### 3.4. Medicine Index & Fast Lookup (`medicine_index.py`)
- Indexes over **503,000 commercial formulations** in `saved_models/medicine_index.pkl`.
- Uses bucketed hash maps by brand aliases, cleaned alphanumeric prefixes, and normalized salt signatures.
- Enables sub-millisecond retrieval without table-scanning large SQLite databases on every request.

### 3.5. Multi-Tier Medicine Matcher (`medicine_matcher.py`)
- Evaluates candidate matches against a **Strict Verification Gate**:
  - Requires **>75% overall confidence** to display as "verified".
  - **Composition Consistency Gate**: Even if the brand name matches, if the detected milligram strength (e.g., 500mg) conflicts with the database entry (e.g., 650mg), the match is rejected or downgraded to unverified to prevent medication errors.
- Produces a granular **Confidence Breakdown**:
  - Brand Score
  - Ingredient Score
  - Strength Score
  - Manufacturer Score
  - Dosage Form Score
  - Overall Weighted Confidence

### 3.6. Bioequivalent Generic Alternatives Engine (`alternative_matcher.py`)
- Queries `medicines.sqlite` for alternatives sharing the **exact same active pharmaceutical ingredients and equivalent strength**.
- Excludes the primary medicine from its own alternatives list.
- Ranks candidates by lowest verified unit price to highlight the **"Cheapest Verified Generic"** with maximum savings.
- Scores manufacturer trust based on regulatory licensing tiers (Tier 1: WHO-GMP manufacturers like Cipla, Sun Pharma, Dr. Reddy's, Torrent, Lupin; Tier 2: CDSCO-licensed national brands).

### 3.7. Offline Price Verifier & Normalization (`price_verifier.py`)
- Normalizes prices between packages of different pack sizes (e.g., a 10-tablet strip vs. a 15-tablet strip vs. a bottle).
- Calculates per-unit (per tablet / per capsule / per ml) price bases so savings comparisons are scientifically accurate.
- Backed by `price_cache.sqlite` and a built-in catalog of standard Indian Maximum Retail Prices (`KNOWN_BRAND_MRPS`).

### 3.8. Clinical Drug Safety & Cross-Interactions (`drug_safety.py`)
- **Side Effects**: Mapped to MedDRA terminology with occurrence frequencies (Very Common, Common, Rare).
- **Drug-Drug Cross-Interactions**: When scanning prescriptions with 2 or more medicines, it computes all unique pairwise combinations $(M_1, M_2)$ and queries `drugs.sqlite` (`drug_interactions` table) to display interaction severity and clinical descriptions.

---

## 4. Local SQLite Database Schemas

### `medicines.sqlite` (503k+ Medicines)
```sql
CREATE TABLE medicines (
    id INTEGER PRIMARY KEY,
    name TEXT,             -- e.g. "Dolo 650 Tablet"
    salt TEXT,             -- e.g. "Paracetamol (650mg)"
    manufacturer TEXT,     -- e.g. "Micro Labs Ltd"
    price REAL,            -- Package MRP in INR (e.g. 33.50)
    source TEXT            -- Data catalog source
);
CREATE INDEX idx_medicines_name ON medicines(name);
CREATE INDEX idx_medicines_salt ON medicines(salt);
```

### `drugs.sqlite` (Clinical Knowledge Base)
```sql
-- Drug-Drug Interactions
CREATE TABLE drug_interactions (
    drug1 TEXT,            -- Active ingredient 1 (canonical)
    drug2 TEXT,            -- Active ingredient 2 (canonical)
    description TEXT,      -- Clinical mechanism and interaction severity
    PRIMARY KEY (drug1, drug2)
);

-- MedDRA Side Effects
CREATE TABLE side_effects (
    drug TEXT,             -- Canonical drug name
    side_effect TEXT,      -- MedDRA term (e.g. "Nausea", "Headache")
    frequency TEXT         -- e.g. "common", "rare", "post-marketing"
);
```

### `price_cache.sqlite` (Persistent Normalized Pricing Cache)
```sql
CREATE TABLE price_verification_cache (
    medicine_id INTEGER PRIMARY KEY,
    medicine_name TEXT,
    mrp REAL,
    package_price REAL,
    unit_price REAL,
    pack_quantity INTEGER,
    pack_unit TEXT,
    price_basis TEXT,
    comparison_eligible INTEGER,
    cached_at TIMESTAMP
);
```

---

## 5. Frontend & UI/UX Design System

- **Design Philosophy**: Clinical-luxury, modern editorial health-tech. Glassmorphic cards with refined shadows, subtle micro-animations, and high contrast for accessibility.
- **Typography**: Google Fonts (`Fraunces` for serif editorial headlines, `Plus Jakarta Sans` for clean data display).
- **Interactivity**:
  - Live Camera Modal: Direct HTML5 `getUserMedia` WebRTC camera stream with front/back camera flipping and real-time canvas capture.
  - Drag-and-Drop Dropzone with client-side thumbnail generation and size validation.
  - Multi-Medicine Selector: Interactive tab switcher allowing users to inspect each medicine in a prescription individually while seeing cross-interaction warnings globally.
  - Scan History Drawer: Fully persistent client-side scan history stored in `localStorage`.

---

## 6. Current Implementation Status

- **100% Offline**: All external cloud APIs (Groq, Gemini, Roboflow, Kaggle, online web scraping) have been completely removed.
- **Zero API Keys Required**: The application runs entirely on `SECRET_KEY` and `DEBUG=True` in `.env`.
- **Verified Accuracy**: Tested against standard Indian medicine strips (e.g., Dolo 650, Shelcal-HD, 1-AL, Calpol, Augmentin).
- **Automated Regression Suite**: 19 automated unit and regression tests passing in `scanner/tests/test_price_verification.py` and `scanner/tests/test_medicine_pipeline.py`.

---

## 7. Concrete Future Roadmap (Discussion Topics for ChatGPT & Next Phase)

When discussing future extensions with ChatGPT or planning subsequent development sprints, consider the following high-impact initiatives:

### 1. Model Quantization & Edge Acceleration (ONNX / OpenVINO / TensorRT)
- **Problem**: Running PyTorch TrOCR on standard low-power CPUs takes ~700ms–1.5s per image.
- **Solution**: Export YOLOv8 and TrOCR to **ONNX INT8 / FP16** or **OpenVINO**. This can reduce inference latency by 3x–5x and cut RAM consumption from ~1.2GB to <350MB, allowing MediScan to run smoothly on edge appliances or inexpensive shared hosting.

### 2. Specialized Indian Prescription Handwriting OCR
- **Problem**: Printed blister packs are easily recognized, but doctors' cursive handwritten prescriptions remain challenging for general OCR models.
- **Solution**: Fine-tune a lightweight Vision Transformer (ViT / TrOCR) specifically on synthetic and annotated Indian medical handwriting datasets (e.g., Doctor Handwriting Prescription datasets from Kaggle / Roboflow).

### 3. Live Edge Video Stream OCR (WebAssembly / OpenCV.js)
- **Problem**: Currently, users take a still photograph and upload it.
- **Solution**: Integrate a client-side WebAssembly YOLO detector in the browser that displays an animated real-time targeting reticle over the medicine strip and automatically snaps the photo when the strip is in sharp focus.

### 4. Patient Medication Reminder & Schedule Assistant
- **Problem**: Patients frequently miss doses or take medications at incorrect times.
- **Solution**: Parse dosage directions (e.g., *"1 tablet BD after food for 5 days"*) using the salt parser and generate downloadable `.ics` calendar events or WhatsApp/SMS reminder schedules.

### 5. Cross-Platform Native Mobile App (Flutter / React Native)
- **Problem**: Desktop and mobile web browsers have variable camera permissions and WebRTC quirks across iOS Safari and Android Chrome.
- **Solution**: Wrap the MediScan offline engine into a native Flutter or React Native mobile application using ONNX Mobile Runtime (CoreML on iOS, NNAPI on Android) for instant, completely offline on-device scanning.

### 6. FHIR / HL7 EHR Standards Integration
- **Problem**: Data extracted by MediScan is confined to the local browser session.
- **Solution**: Export scanned prescriptions into standardized **HL7 FHIR (Fast Healthcare Interoperability Resources)** JSON bundles (`MedicationRequest`, `MedicationStatement`), allowing integration with hospital information systems and digital health lockers (such as India's Ayushman Bharat Digital Mission - ABDM).

### 7. 2D DataMatrix & GS1 Barcode Decoding
- **Problem**: Faded foil prints can be difficult to read with pure OCR.
- **Solution**: Add a complementary barcode reader (using `pyzbar` or `zxing-cpp`) to decode GS1 DataMatrix barcodes printed on newer Indian pharmaceutical strips for instant 100% verified GTIN, batch number, and expiry date identification.

### 8. Chronic Prescription Basket Cost Optimizer
- **Problem**: Patients with chronic conditions (diabetes, hypertension, cardiac care) purchase 4 to 8 medicines monthly.
- **Solution**: Build a "Monthly Basket Calculator" that optimizes the entire combination of medications simultaneously, showing total monthly savings and grouping alternatives from single reputable manufacturers to minimize delivery and pharmacy friction.

### 9. Voice & Multilingual Vernacular Accessibility
- **Problem**: Many elderly or rural patients may not be comfortable reading English medical terminology.
- **Solution**: Integrate offline Text-to-Speech (TTS) and multilingual translation (Hindi, Tamil, Telugu, Malayalam, Bengali, Marathi) to speak out the medicine name, purpose, dosage instructions, and critical safety warnings aloud.

### 10. Vector Semantic Search for Phonetic Drug Name Matching
- **Problem**: Extreme OCR typos (e.g., "M1cr0-D0l0" or smudged blister foil) can miss exact alias matching.
- **Solution**: Implement local SQLite vector search (`sqlite-vss`) with lightweight character n-gram embeddings or phonetic encoders (Double Metaphone / Soundex) to bridge severe character recognition degradation.
