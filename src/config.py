# src/config.py
import os
from pathlib import Path

# --- ENVIRONMENT DETECTION ---
IS_KAGGLE = os.path.exists("/kaggle")

if IS_KAGGLE:
    # Updated to match Kaggle's nested folder structure
    RAW_DATA_DIR = Path("/kaggle/input/datasets/mufeedkeenari/lbnl-hrtem-dataset")
    PROCESSED_DIR = Path("/kaggle/working")
else:
    # On my system
    RAW_DATA_DIR = Path("data/raw")
    PROCESSED_DIR = Path("data/processed")

# Ensure the output directory always exists so saving doesn't crash
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- FILE NAMES ---
# Centralizing filenames prevents typos across different scripts
IMAGES_FILENAME = "au_10nm_images.h5"
MASKS_FILENAME = "au_10nm_labels.h5"
WEIGHTS_FILENAME = "unet_tem_weights.pth"