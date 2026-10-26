# MediScan

MediScan is a local Django application for reading medicine information from two kinds of images. The scan mode is selected by the user and does not change automatically based on image shape.

## Scan modes

- **Printed Labels** reads printed text using EasyOCR text-region detection and the printed TrOCR model.
- **Handwritten Labels** reads handwriting with a separate OCR pipeline.

Both modes match recognized medicine names and ingredients against the local medicine database. Printed-label results distinguish **Verified**, **Likely**, and **Composition match** outcomes. The result page can show ingredients, manufacturer, safety information, and available same-composition alternatives. Verified price comparisons are shown only when the price and package basis can be compared.

## How scanning works

1. The user selects Printed Labels or Handwritten Labels and uploads an image.
2. Django saves the upload under `media/uploads/` and calls the selected scan pipeline.
3. OCR reads text regions. The medicine parser extracts names, active ingredients, and strengths.
4. The matcher compares the read brand and exact active-ingredient composition against the local index and SQLite database.
5. Django stores the result in the session and renders the report page.

The printed and handwritten recognizers are loaded once and reused. When started with Django's development server, MediScan warms the printed recognizer in the background. The first startup can take longer while the local model loads; later scans reuse it.

## Main project files

```text
mediscan/                  Django project settings and URL configuration
scanner/views.py            Upload, scan, session, and result views
scanner/ml/                 OCR, text extraction, matching, alternatives, and safety lookup
scanner/ml/printed_trocr.py Printed-label OCR
scanner/ml/handwritten_trocr.py Handwritten-note OCR
scanner/ml/medicine_matcher.py Brand/composition decisions and alternatives
scanner/ml/medicine_index.py Local medicine database and search index
templates/scanner/          Upload and report pages
static/                     CSS, JavaScript, and image assets
saved_models/               Local OCR models, databases, and generated index
media/uploads/              Temporary uploaded scans
test_images/                Local sample images
tools/                      Regression, evaluation, and development utilities
```

Model weights, local medicine databases, and generated index files are not all committed to Git. The local environment needs the files configured in `mediscan/settings.py`, including:

- `saved_models/modelA_printed_ocr/` for printed labels
- `saved_models/modelB_handwritten_ocr/` for handwritten notes
- `saved_models/medicines.sqlite` and `saved_models/drugs.sqlite`
- `saved_models/medicine_index.pkl` (built or loaded by the medicine index)

The optional package-cropping setting is `USE_STRIP_CROP`; it is disabled by default.

## Setup

Use Python 3.11 and install the dependencies in `requirements.txt` in an environment compatible with the installed PyTorch build. Provide the local model and database files listed above, then configure `.env` from `.env.example`.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000/`. On first launch, allow the OCR model warm-up to finish before scanning. The local development server prints its runtime diagnostics during startup.

## Checks and sample images

```powershell
python manage.py check
python manage.py test scanner.tests
python tools/test_live_regression.py
```

`tools/test_live_regression.py` uploads the regression photos through Django's test client and checks the rendered result page. The test images are under `test_images/` and `tools/test_photos/`. These checks require the local models and medicine databases.

## Local data and privacy

Scans run using local models and databases. Uploaded images are stored under `media/uploads/`; the upload handler removes older upload files during later uploads. Keep `.env`, local databases, model weights, and user images private.
