"""
config.py  -  the ONLY file where you edit paths/settings.
Every other script imports from here.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------
# 1. WHERE YOUR BreaKHis (binary) FOLDER IS
#    Expected layout:  DATASET_ROOT / 40X / benign / *.png   (same for 100X, 200X, 400X, malignant)
# ---------------------------------------------------------------
DATASET_ROOT = Path(os.environ.get(
    "BREAKHIS_ROOT",
    r"C:\Users\layam\Desktop\capston\dataset_cancer_v1\dataset_cancer_v1\classificacao_binaria",
))

# ---------------------------------------------------------------
# 2. OUTPUT FOLDERS (created automatically next to these scripts)
# ---------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"          # split csv files
CKPT_DIR    = PROJECT_DIR / "checkpoints"   # trained model weights
RESULTS_DIR = PROJECT_DIR / "results"       # predictions, tables, plots
CACHE_DIR   = PROJECT_DIR / "cache"         # pre-resized images (made by cache_images.py)
for _d in (DATA_DIR, CKPT_DIR, RESULTS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------
# 3. FIXED PROJECT SETTINGS
# ---------------------------------------------------------------
MAGS    = ["40X", "100X", "200X", "400X"]   # ESCALATION ORDER: cheapest first
CLASSES = ["benign", "malignant"]           # label 0 = benign, 1 = malignant
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

BACKBONE = "convnextv2_atto.fcmae_ft_in1k"  # same encoder family as before
IMG_SIZE = 224
