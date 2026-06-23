# MediScan — Visual Medicine Finder

## Setup Instructions

### 1. Clone and install
git clone https://github.com/karthikunnikrishnan/mediscan.git
cd mediscan
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

### 2. Download model weights
from huggingface_hub import hf_hub_download, snapshot_download

# Strip detector
snapshot_download(repo_id="karthikunnikrishnan/mediscan-strip-detector",
                  local_dir="saved_models/")

# Prescription HTR
snapshot_download(repo_id="karthikunnikrishnan/mediscan-prescription-htr",
                  local_dir="saved_models/prescription_htr/")

### 3. Build databases
Download datasets from Kaggle (links in training scripts)
python training/build_medicine_db.py
python training/build_drug_db.py

### 4. Run
python manage.py migrate
python manage.py runserver
