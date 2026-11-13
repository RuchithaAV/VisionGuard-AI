# VisionGuard AI - Industrial Visual Quality Inspection & Defect Detection System

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Framework: TensorFlow/Keras](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-orange.svg)](https://tensorflow.org/)
[![Dataset: MVTec AD](https://img.shields.io/badge/Dataset-MVTec%20AD-green.svg)](https://www.mvtec.com/company/research/datasets/mvtec-ad)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

> **VisionGuard AI** is an industry-oriented deep learning and computer vision system engineered for automated, unsupervised visual quality inspection. It detects anomalous product surfaces, calculates calibrated anomaly scores, localizes pixel-level defects via heatmaps, and renders automated factory decisions (**PASS**, **FAIL**, **REVIEW**).

---

## 1. Problem Statement

In modern smart manufacturing, visual quality inspection is critical to ensure zero-defect outgoing quality. However, traditional inspection faces major hurdles:
- **Human Inspection Limitations:** Prone to fatigue, subjective bias, high cost, and inability to scale at production line speeds.
- **The Imbalance Challenge:** Defective samples are rare, diverse, and practically unpredictable. Standard supervised classification fails because defective training examples are scarce, and new unseen defect types constantly emerge.
- **Microscopic Defect Footprints:** As verified in our exploratory benchmarks, industrial defects often occupy **less than 0.25% of the total product area** (e.g., hairline cracks and thread scratches). Global pooling classifiers wash out these subtle signals.

VisionGuard AI solves this using **unsupervised representation learning**, training exclusively on nominal (defect-free) product images to learn the structural manifold of healthy parts and flagging deviations.

---

## 2. Technical Architecture & Multi-Model Approach

VisionGuard AI implements two complementary anomaly detection architectures alongside a calibrated decision engine:

```
                          ┌──────────────────────────┐
                          │ Incoming Industrial Image │
                          └─────────────┬────────────┘
                                        │
                         [ Preprocessing & Resizing ]
                         [ (256x256x3), INTER_AREA  ]
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
    ┌──────────────────────────┐                  ┌──────────────────────────┐
    │       MODEL 1            │                  │       MODEL 2            │
    │  Custom Conv Autoencoder │                  │ PatchCore Feature-Memory │
    │  (Unsupervised Recon.)   │                  │ (Pretrained EfficientNet)│
    └────────────┬─────────────┘                  └────────────┬─────────────┘
                 │                                             │
      Reconstruction Error                          Nearest-Neighbor Patch
       Map (|X - X_hat|)                             Distance in Feature Space
                 │                                             │
                 └──────────────────────┬──────────────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │   Anomaly Analysis Engine   │
                         │  • Pixel Heatmap & Overlay  │
                         │  • Image-level Anomaly Score│
                         │  • Defect Footprint Area %  │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │  Calibrated Decision Engine │
                         │  • PASS                     │
                         │  • REVIEW                   │
                         │  • FAIL                     │
                         └─────────────────────────────┘
```

### Model 1: Custom Convolutional Autoencoder
- **Principle:** Learns a compressed latent representation of nominal textures and geometries. When an anomalous image is presented, the decoder fails to reconstruct the defect, generating high reconstruction error.
- **Loss Strategy:** Combined Pixel Reconstruction Loss (MAE/MSE) and Structural Similarity (SSIM) to counteract dark background dominance.

### Model 2: PatchCore-Inspired Feature Memory Bank
- **Principle:** Extracts multi-scale intermediate feature maps from an ImageNet-pretrained **EfficientNetB0** backbone.
- **Mechanism:** Builds a core-set memory bank of nominal patch embeddings. At test time, test patches are scored by their nearest-neighbor Euclidean distance against normal reference patches, yielding precise, high-resolution anomaly heatmaps.

### Anomaly Analysis & Decision Engine
- **Outputs:** Continuous anomaly score $S \in [0, 1]$, binary anomaly segmentation mask, and defect footprint percentage.
- **Decision Boundaries:**
  - **`PASS`**: $S \le \tau_{\text{pass}}$ (Nominal product, no significant deviation).
  - **`REVIEW`**: $\tau_{\text{pass}} < S \le \tau_{\text{fail}}$ (Uncertain prediction; flagged for human QA verification).
  - **`FAIL`**: $S > \tau_{\text{fail}}$ (Confirmed defect; rejected from line).

---

## 3. Dataset & Empirical Discoveries

VisionGuard AI utilizes the benchmark **MVTec Anomaly Detection (MVTec AD)** dataset, focusing on three structurally distinct industrial categories:

| Category | Structural Profile | Raw Resolution | Native Mode | Defect Types | Defect Footprint (% of image) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`metal_nut`** | Rigid symmetric metal ring with threading | $700 \times 700$ | RGB | `bent`, `color`, `flip`, `scratch` | $0.436\% - 48.439\%$ |
| **`screw`** | Elongated threaded mechanical fastener | $1024 \times 1024$ | Grayscale (`L`) | `manipulated_front`, `scratch_head`, `scratch_neck`, `thread_side`, `thread_top` | **$0.151\% - 0.649\%$** |
| **`capsule`** | Non-metallic pharmaceutical pill | $1000 \times 1000$ | RGB | `crack`, `faulty_imprint`, `poke`, `scratch`, `squeeze` | **$0.249\% - 7.694\%$** |

### Verified Exploratory Findings:
1. **Grayscale vs. Color:** `screw` images are natively single-channel grayscale (`Mode: L`), while `metal_nut` and `capsule` are 3-channel RGB. The preprocessing pipeline standardizes all inputs to 3-channel RGB to satisfy pretrained backbone contracts.
2. **Microscopic Defect Scales:** In `screw`, the `thread_side` defect occupies merely **$0.151\%$** of the total image (1,580 out of 1,048,576 pixels). In `capsule`, hairline `crack` defects occupy **$0.249\%$**.
3. **Interpolation Artifacts:** Quantization tests confirmed that downsampling binary masks with bilinear interpolation introduces non-binary intermediate artifacts (`[0, 64, 128, 191, 255]`), whereas `INTER_NEAREST` preserves strict binary truth (`[0, 255]`).

---

## 4. Repository Structure

```text
VisionGuard-AI/
├── configs/
│   ├── data.yaml              # Centralized data paths, image sizing & splits
│   ├── model.yaml             # Model hyperparameters & backbone settings
│   └── training.yaml          # Learning rates, batch sizes, optimizers
│
├── data/
│   ├── raw/                   # MVTec AD categories (git-ignored)
│   ├── processed/             # Standardized TFRecords / caches
│   └── README.md              # Dataset acquisition & licensing guidelines
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Validated exploratory data analysis
│   ├── 02_preprocessing.ipynb       # Pipeline prototyping
│   ├── 03_cnn_autoencoder.ipynb     # Autoencoder baseline & experiments
│   ├── 04_efficientnet.ipynb        # PatchCore feature-memory experiments
│   └── 05_model_comparison.ipynb    # Comprehensive benchmarking & metrics
│
├── src/
│   ├── data/
│   │   ├── download_mvtec.py  # Automated verified dataset fetcher
│   │   ├── verify_dataset.py  # Automated directory & mask integrity verifier
│   │   ├── preprocessing.py   # Image normalization & mask binarization
│   │   └── loader.py          # High-performance tf.data streaming pipelines
│   │
│   ├── models/
│   │   ├── autoencoder.py     # Functional Keras Convolutional Autoencoder
│   │   ├── efficientnet.py    # Pretrained multi-scale feature extractor
│   │   └── feature_memory.py  # Core-set sampling & nearest-neighbor search
│   │
│   ├── training/
│   │   ├── train_autoencoder.py # GradientTape custom training loops
│   │   └── train_utils.py       # Learning rate schedulers & early stopping
│   │
│   ├── inference/
│   │   ├── anomaly_detector.py # End-to-end inference orchestrator
│   │   ├── localization.py     # Heatmap generation & mask thresholding
│   │   └── decision_engine.py  # Calibrated PASS / REVIEW / FAIL classifier
│   │
│   ├── evaluation/
│   │   ├── metrics.py          # Image-AUROC, Pixel-AUROC, IoU, AUPRO
│   │   └── visualization.py    # Inspection overlay and error reporting
│   │
│   └── utils/
│       ├── logger.py           # Structured logging
│       └── config.py           # Configuration loader
│
├── api/                       # FastAPI REST service (Milestone 07)
├── dashboard/                 # Streamlit operational inspection UI (Milestone 07)
├── tests/                     # Pytest test suite (Milestone 08)
├── Dockerfile                 # Containerization specification (Milestone 08)
├── requirements.txt           # Pinned production dependencies
└── README.md
```

---

## 5. Development Roadmap & Milestones

| Milestone | Focus | Key Deliverables | Status |
| :---: | :--- | :--- | :---: |
| **01** | **Dataset & Data Engineering** | Automated verification, exploratory notebook, verified TF pipeline | 🟢 *Completed (Verified)* |
| **02** | **Convolutional Autoencoder** | Encoder-decoder functional model, MSE/SSIM loss, reconstruction maps | 🟢 *Completed (Verified)* |
| **03** | **Advanced Autoencoder Studies** | Custom `tf.GradientTape` loops, bottleneck tuning, threshold calibration | 🟡 *In Progress* |
| **04** | **PatchCore Feature Memory** | EfficientNetB0 feature extraction, memory bank, nearest-neighbor scoring | ⚪ *Planned* |
| **05** | **Evaluation & Benchmarking** | Image-AUROC, Pixel-AUROC, IoU, AUPRO, MLflow tracking | ⚪ *Planned* |
| **06** | **Decision Engine & XAI** | Defect area % estimation, heatmap blending, PASS/FAIL/REVIEW logic | ⚪ *Planned* |
| **07** | **Application Interface** | FastAPI REST endpoints, interactive Streamlit inspection dashboard | ⚪ *Planned* |
| **08** | **Production Engineering** | Docker containerization, unit/integration testing (pytest), CI/CD | ⚪ *Planned* |
| **09** | **Domain Shift & Portfolio** | Generalization testing on VisA dataset, technical reporting | ⚪ *Planned* |

---

## 6. Getting Started

### 1. Clone & Set Up Environment
```powershell
git clone https://github.com/RuchithaAV/VisionGuard-AI.git
cd VisionGuard-AI

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Dataset Integrity
```powershell
python src/data/verify_dataset.py
```
Outputs automated integrity validation across all train, test, and ground-truth mask directories.

### 3. Run Exploration Notebook
Open `notebooks/01_data_exploration.ipynb` in Antigravity / VS Code with the `.venv` kernel to view defect footprints, channel distributions, and downscaling benchmarks.

### 4. Run Automated Unit Tests (9 Tests Passing)
```powershell
python -m pytest -v
```
Verifies image preprocessing ranges, strict binary mask outputs, zero-leakage dataset splits, Keras functional layer dimensions, forward-pass bounds, and inference localization math.

---

## 7. Engineering & Evaluation Protocol

- **Strict Zero Data Leakage:** Training is conducted exclusively on nominal (`train/good`) samples. Validation splits for loss monitoring and threshold calibration are carved strictly from the nominal training pool. The official `test/` directory is touched only during final benchmarking.
- **Beyond Accuracy:** Models are evaluated on **Image-AUROC**, **Pixel-AUROC**, **Intersection over Union (IoU)**, and **Area Under the Per-Region Overlap (AUPRO)** curve, paired with inference latency measurements ($ms/\text{image}$).

---

## 8. License & Attribution

- **Code:** Licensed under the MIT License.
- **Dataset:** MVTec AD is provided by MVTec Software GmbH under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/). Strictly intended for non-commercial research and educational demonstration.
- **Author:** **Ruchitha Vithana** — BSc (Hons) in Information Technology & Management, University of Colombo.
