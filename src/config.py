# src/config.py
import os
from pathlib import Path

# --- ENVIRONMENT DETECTION ---
# Kaggle's cloud environment always has a /kaggle directory.
# Your local Windows machine does not.
IS_KAGGLE = os.path.exists("/kaggle")

if IS_KAGGLE:
    # On Kaggle, datasets are mounted in /kaggle/input/<your-dataset-name>
    RAW_DATA_DIR = Path("/kaggle/input/lbnl-hrtem-dataset")
    # /kaggle/working/ is the only place Kaggle lets you save files
    PROCESSED_DIR = Path("/kaggle/working")
else:
    # On your local PyCharm machine, use standard relative paths
    RAW_DATA_DIR = Path("data/raw")
    PROCESSED_DIR = Path("data/processed")

# Ensure the output directory always exists so saving doesn't crash
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- FILE NAMES ---
# Centralizing filenames prevents typos across different scripts
IMAGES_FILENAME = "au_10nm_images.h5"
MASKS_FILENAME = "au_10nm_labels.h5"
WEIGHTS_FILENAME = "unet_tem_weights.pth"