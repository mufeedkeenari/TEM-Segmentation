# TEM Image Segmentation Engine

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Materials Informatics](https://img.shields.io/badge/Domain-Materials_Informatics-8A2BE2.svg)]()

A production-grade Computer Vision pipeline for the automated segmentation of nanoparticles in High-Resolution Transmission Electron Microscopy (HRTEM) imagery. 

This project aims to bridge the gap between experimental materials characterization and high-throughput computational analysis by replacing manual, error-prone nanoparticle sizing with a robust Deep Learning architecture.

## 🔬 Dataset & Data Engineering
This engine is trained on the public, peer-reviewed [Lawrence Berkeley National Laboratory (LBNL) Segmented HRTEM Nanoparticle Dataset](https://datadryad.org/). It is currently optimized for the **Au 10nm 330kx** subset.

**Engineering Highlights:**
* **Lazy-Loading HDF5 Architecture:** Standard `.h5` file ingestion often causes system deadlocks when combined with PyTorch's `DataLoader` multiprocessing. This pipeline utilizes a strictly typed, dual-file lazy-loading `Dataset` class to dynamically stream tensors into memory, ensuring thread safety and scalable training.
* **Geometric Augmentation:** TEM data is orientation-agnostic. The pipeline integrates `Albumentations` to inject real-time physical variance (rotations, flips, scaling), artificially expanding the dataset and preventing model overfitting.
* **Automated Mathematical Verification:** Features a comprehensive `pytest` suite utilizing mocked HDF5 matrices to assert tensor geometries (`CxHxW`) and categorical normalizations before forward-pass execution.

## 🏗️ Project Architecture

```text
TEM-Segmentation/
├── data/
│   ├── raw/               # Immutable LBNL HDF5 physical data
│   └── processed/         # Pipeline outputs
├── notebooks/
│   └── 01_eda.ipynb       # Visual verification of human-annotated masks
├── src/
│   ├── data/
│   │   ├── dataset.py     # Dual-h5 ingestion engine
│   │   └── transforms.py  # Albumentations CV augmentation logic
│   ├── models/
│   │   └── unet.py        # [WIP] Neural network architecture
│   ├── engine/
│   │   ├── loss.py        # [WIP] Objective functions
│   │   └── train.py       # [WIP] Main training loop
│   └── inference/
│       └── predict.py     # [WIP] High-throughput inference script
├── tests/
│   └── test_dataset.py    # Pytest sanity checks for tensor bounds
├── pyproject.toml         # Environment configurations
└── requirements.txt       # Frozen dependencies

```

## 🚀 Getting Started

**1. Clone and Configure**

```bash
git clone https://github.com/YOUR_USERNAME/TEM-Segmentation.git
cd TEM-Segmentation
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -r requirements.txt

```

**2. Run the Test Suite**
Verify the ingestion engine is mathematically sound on your local hardware:

```bash
pytest tests/

```

**3. Visual Inspection**
Launch Jupyter to verify mask alignment on the physical TEM data:

```bash
jupyter notebook notebooks/01_eda.ipynb

```

## 📈 Roadmap

* [x] **Phase 1:** Scaffold modular architecture and strict `.gitignore` protocol.
* [x] **Phase 2:** Implement lazy-loading HDF5 data ingestion and Pytest verification.
* [x] **Phase 3:** Integrate CV augmentation and perform EDA mask alignment.
* [ ] **Phase 4:** Construct PyTorch U-Net architecture.
* [ ] **Phase 5:** Implement custom Dice/IoU loss functions and standard training loops.
* [ ] **Phase 6:** Export inference engine for new, unlabelled TEM micrographs.


