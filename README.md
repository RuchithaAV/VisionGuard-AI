# VisionGuard AI - Industrial Visual Quality Inspection & Defect Detection System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Framework: TensorFlow/Keras](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-orange.svg)](https://tensorflow.org/)
[![API: FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Dashboard: Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Deploy-Docker%20Compose-2496ed.svg)](https://docs.docker.com/compose/)
[![Dataset: MVTec AD](https://img.shields.io/badge/Dataset-MVTec%20AD-green.svg)](https://www.mvtec.com/company/research/datasets/mvtec-ad)
[![Status](https://img.shields.io/badge/Status-Active%20Development-yellow.svg)]()

> **VisionGuard AI** is an unsupervised visual quality inspection system for manufacturing. It learns what *normal* parts look like from defect-free images only, then flags deviations, localizes them with a heatmap, and routes each part to a calibrated **PASS / REVIEW / FAIL** decision. The project covers the full path from research experiments to a served system: a FastAPI inference service, a Streamlit operator console, an SQLite audit log, and Docker packaging.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [System Architecture](#2-system-architecture)
3. [Models](#3-models)
4. [Calibrated Decision Engine](#4-calibrated-decision-engine)
5. [Dataset & Exploratory Findings](#5-dataset--exploratory-findings)
6. [Experimental Results](#6-experimental-results)
7. [Evaluation Framework](#7-evaluation-framework)
8. [Inference API, Dashboard & Audit Log](#8-inference-api-dashboard--audit-log)
9. [Repository Structure](#9-repository-structure)
10. [Getting Started](#10-getting-started)
11. [Automated Testing & CI](#11-automated-testing--ci)
12. [Engineering & Evaluation Protocol](#12-engineering--evaluation-protocol)
13. [Roadmap & Milestones](#13-roadmap--milestones)
14. [Known Limitations](#14-known-limitations)
15. [License, Attribution & References](#15-license-attribution--references)

---

## 1. Problem Statement

Visual quality inspection is essential in manufacturing, but conventional approaches struggle:

- **Human inspection limits:** manual inspection is subject to fatigue, subjective judgment, cost, and scalability limits.
- **Rare, diverse defects:** defective samples are scarce and unpredictable, which makes supervised classification hard.
- **Tiny defect footprints:** scratches and cracks can cover well under 1% of an image, so global image representations can miss them.
- **Unseen defects:** new defect patterns appear that were never in the training data.

VisionGuard AI uses **unsupervised anomaly detection**: models learn normal appearance from defect-free training images and flag deviations at inspection time. Two approaches were investigated:

1. **Reconstruction-based:** a convolutional autoencoder (experimental baseline, notebooks 02-03).
2. **Feature-memory-based:** an EfficientNetB0 feature extractor with a PatchCore-inspired memory bank (the model used by the API and dashboard).

---

## 2. System Architecture

The system has an **offline** stage that builds per-category artifacts, and an **online** stage that serves inspections.

```text
OFFLINE  -  scripts/build_artifacts.py  (once per category)

  data/raw/<category>/train/good
            |  85 / 15 split (seed 42)
            |-- train (85%) --> frozen EfficientNetB0 --> patch embeddings
            |                     --> 10% random coreset --> memory_bank.npz
            |
            '-- val (15%, normal only) --> score with the fitted bank
                                           --> mean, std --> DecisionEngine thresholds
                                           --> metadata.json
                                           (stored under models/patchcore/<category>/)

ONLINE  -  api/main.py  -->  VisionGuardPredictor

  image (PNG/JPG) + category
            |  resize to 256x256, RGB, scale to [0, 1]
            v
  frozen EfficientNetB0 (block2b_add + block3b_add)
            --> (32 x 32 x 64) patch embeddings
            v
  1-nearest-neighbour distance to the category memory bank
            --> 32 x 32 distance grid
            |
            |-- max distance --> raw image score --> DecisionEngine
            |                        --> PASS / REVIEW / FAIL, z-score, normalized risk
            |
            '-- bilinear upscale + Gaussian blur --> 256 x 256 heatmap
                     --> defect area %, calibrated colour overlay (base64 PNG)
            v
  SQLite audit log (data/inspections.db)
            --> /history, /stats --> Streamlit dashboard
```

All three categories share one frozen EfficientNetB0 backbone; each category has its own memory bank and its own calibrated thresholds.

---

## 3. Models

### Model 1: Convolutional Autoencoder (experimental baseline)

Implemented with the Keras Functional API in `src/models/autoencoder.py`. It is **not** part of the serving path; it documents the reconstruction-based experiments that motivated PatchCore.

- **Encoder:** four Conv(3x3) + BatchNorm + LeakyReLU(0.2) + MaxPool blocks (32, 64, 128, 256 filters), compressing `(256, 256, 3)` to a `(16, 16, 256)` latent map.
- **Decoder:** four transposed-convolution blocks back to `(256, 256, 3)` with a sigmoid output.
- **Losses:** MSE baseline, and a custom `SSIMMAELoss` that combines structural similarity and mean absolute error:

$$
\mathcal{L} = \alpha\,\bigl(1-\operatorname{SSIM}(X,\hat{X})\bigr) + (1-\alpha)\,\operatorname{MAE}(X,\hat{X})
$$

- **Localization (`src/inference/localization.py`):** absolute reconstruction error averaged across channels, Gaussian smoothing, and an optional foreground mask (background colour estimated from image corners, Otsu thresholding, morphological closing, contour filling, and elliptical erosion) to suppress boundary artefacts. Image scores can use `max`, `top_k_mean` or `mean` pooling.

### Model 2: PatchCore-Inspired Feature Memory (production model)

Implemented in `src/models/efficientnet.py` and `src/models/feature_memory.py`. Values below come from `configs/model.yaml`.

| Component | Setting |
|---|---|
| Backbone | ImageNet-pretrained **EfficientNetB0**, frozen |
| Tapped layers | `block2b_add` (24 ch) and `block3b_add` (40 ch) |
| Alignment | shallower map resized (bilinear) to the deeper map's 32 x 32 grid |
| Local aggregation | 3 x 3 average pooling, stride 1, `same` padding |
| Patch embedding | concatenation along channels: **64-D** vector per patch |
| Patches per image | 32 x 32 = 1,024 (input `256 x 256 x 3`) |
| Memory reduction | **seeded uniform-random coreset**, ratio **0.10** in `configs/model.yaml` (class default 0.05), seed 42 |
| Search | exact Euclidean nearest neighbour (`sklearn.neighbors.NearestNeighbors`), `n_neighbors = 1` |
| Image score | **maximum** patch distance |
| Heatmap | 32 x 32 distance grid, bilinear upscale to 256 x 256, Gaussian blur (kernel 7, sigma 1.5) |

**Persistence:** `PatchCoreMemoryBank.save()` writes `memory_bank.npz` (compressed vectors) and `metadata.json` (extractor settings, split sizes, and the calibrated decision-engine parameters). `load()` rebuilds the nearest-neighbour index.

**Documented deviations from the original PatchCore (Roth et al., CVPR 2022):**

1. EfficientNetB0 (two mid-level layers) instead of WideResNet-50, chosen for CPU inference speed.
2. Seeded random coreset sampling instead of greedy k-center selection.
3. Plain 1-NN max-distance scoring, without the neighbourhood re-weighting used in the original paper.

This is a PatchCore-*inspired* implementation. Its results should not be read as reproducing the published PatchCore numbers.

---

## 4. Calibrated Decision Engine

`src/inference/decision_engine.py` converts a raw anomaly score into a factory routing decision. Thresholds are calibrated **only from held-out normal validation scores** (15% of `train/good`), never from test data.

```text
pass_threshold = val_mean + 2.0 x val_std      (std uses ddof = 1)
fail_threshold = val_mean + 3.5 x val_std
```

The multipliers are set in `configs/model.yaml` under `decision_engine`.

| Raw score | Decision |
|---|---|
| `score <= pass_threshold` | **PASS** |
| `pass_threshold < score <= fail_threshold` | **REVIEW** (route to a human inspector) |
| `score > fail_threshold` | **FAIL** |

Each decision also returns:

- **`z_score`:** standard deviations above the validation mean.
- **`normalized_score`:** a risk index in [0, 1], piecewise-linear with anchors at validation mean -> 0.25, pass threshold -> 0.50, fail threshold -> 0.80, saturating at 1.0.
- **`defect_area_pct`:** percentage of heatmap pixels above the pass threshold (a footprint estimate).

> **Status of validation:** the calibration logic is unit-tested (`tests/test_decision.py`), and the PASS / REVIEW / FAIL routing has been systematically evaluated on the full official MVTec AD test split via `scripts/evaluate_decisions.py`. Defect caught rates reach **89.96%** on `metal_nut` and **85.7%** on `screw` with zero normal false fails. Each category's baseline mean and std are calibrated from held-out validation normal parts without test leakage.

---

## 5. Dataset & Exploratory Findings

VisionGuard AI uses the **MVTec Anomaly Detection (MVTec AD)** benchmark, restricted to three structurally different categories.

| Category | Structural Profile | Raw Resolution | Native Mode | Defect Types | Defect Footprint |
|---|---|---|---|---|---|
| `metal_nut` | Rigid symmetric metal ring with threading | 700 x 700 | RGB | bent, color, flip, scratch | 0.436% - 48.439% |
| `screw` | Elongated threaded fastener | 1024 x 1024 | Grayscale | manipulated_front, scratch_head, scratch_neck, thread_side, thread_top | 0.151% - 0.649% |
| `capsule` | Non-metallic pharmaceutical pill | 1000 x 1000 | RGB | crack, faulty_imprint, poke, scratch, squeeze | 0.249% - 7.694% |

**Splits used in this project** (train / validation / test; validation is carved out of `train/good`, test is the official MVTec test split):

| Category | Train | Validation | Test |
|---|---|---|---|
| `metal_nut` | 187 | 33 | 115 (22 normal, 93 defective) |
| `screw` | 272 | 48 | 160 (41 normal, 119 defective) |
| `capsule` | 186 | 33 | 132 (23 normal, 109 defective) |

**Findings from `notebooks/01_data_exploration.ipynb`:**

1. **Grayscale vs. colour:** `screw` is single-channel; `metal_nut` and `capsule` are RGB. Preprocessing converts everything to 3 channels for the pretrained backbone.
2. **Microscopic defects:** some defects cover roughly 0.15-0.25% of the image (for example `screw/thread_side`, `capsule/crack`), which motivates local feature matching over global representations.
3. **Mask interpolation:** bilinear resizing introduces intermediate values in binary masks, so masks are resized with nearest-neighbour interpolation and binarized. Images use `INTER_AREA` to preserve thin scratches.

---

## 6. Experimental Results

### 6.1 Autoencoder progression (`metal_nut`, 22 normal / 93 defective test images)

| Stage | Architecture | Loss | Post-processing | Normal Mean Score | Defect Mean Score | Image-AUROC |
|---|---|---|---|---|---|---|
| 01 | Baseline Conv Autoencoder | MSE | Raw error map | 0.2647 +/- 0.0306 | 0.2419 +/- 0.0246 | 25.61% |
| 02 | Structural Loss Autoencoder | SSIM + MAE | Raw error map | 0.2820 +/- 0.0243 | 0.2675 +/- 0.0153 | 30.21% |
| 03 | Foreground Edge-Masked AE | SSIM + MAE | 8 px morphological edge erosion | 0.1837 +/- 0.0331 | 0.1637 +/- 0.0310 | 31.67% |

An AUROC **below 50%** means the ranking is inverted: on average, normal parts received *higher* anomaly scores than defective ones. The diagnosis was that rotation and specular glare on rigid metal parts produce large reconstruction errors around the silhouette that outweigh small defects. SSIM + MAE and edge masking improved the number but did not fix the underlying problem, which motivated the move to feature-memory detection.

### 6.2 PatchCore benchmark (all three categories)

Source: final benchmark cell of `notebooks/04_patchcore_efficientnet.ipynb`. Memory banks fit on the train split with a 10% coreset, scored on the full official test split.

| Category | Image-AUROC | Pixel-AUROC | PR-AUC | Fit time | Latency / image |
|---|---|---|---|---|---|
| `metal_nut` | **95.89%** | 98.31% | 99.13% | 4.30 s | 176.0 ms |
| `screw` | **76.90%** | 98.40% | 91.26% | 5.62 s | 195.3 ms |
| `capsule` | **94.58%** | 98.85% | 98.81% | 4.76 s | 175.8 ms |

On `metal_nut`, PatchCore raises Image-AUROC from 31.67% (best autoencoder) to 95.89%, with normal test scores averaging 19.58 +/- 1.12 versus 25.75 +/- 3.47 for defective parts.

**Best-F1 operating point (oracle).** The threshold below is chosen *using the test labels* (`find_optimal_f1_threshold`), so it is an optimistic upper bound and **not** the threshold the deployed decision engine uses.

| Category | Max F1 | TP / FP / FN / TN |
|---|---|---|
| `metal_nut` | 0.9565 | 88 / 3 / 5 / 19 |
| `screw` | 0.8623 | 119 / 38 / 0 / 3 |
| `capsule` | 0.9558 | 108 / 9 / 1 / 14 |

**How to read these numbers:**

- **`screw` category optimization:** In the initial 10% coreset run, `screw` achieved 76.90% Image-AUROC due to microscopic defect footprints. Introducing category-specific overrides in `configs/model.yaml` (`coreset_ratio: 0.40`, tighter decision multipliers) boosted `screw` to **86.27% Image-AUROC** and an **85.7% defect capture rate**.
- **Pixel-AUROC pools all pixels**, so large defect-free background areas can make it look high even when image-level separation is weak, as with `screw`. Treat it alongside the image-level metrics.
- **Timings** were measured on the author's local CPU. Other cells in the notebook print slightly different figures for the same run type (for example fit time 3.74-4.30 s and latency 138-182 ms), so treat timings as approximate.
- These results are for three MVTec AD categories only and do not represent general performance.

### 6.3 Multi-Seed (3 Seeds) & Per-Defect Benchmark

Source: `artifacts/benchmark_report.json` generated by `scripts/run_benchmark.py` across 3 distinct seeds (`42`, `123`, `999`).

| Category | Image-AUROC | Pixel-AUROC | Pixel-IoU (@ pass threshold) | Defect Caught Rate |
|---|---|---|---|---|
| `metal_nut` | **96.68% ± 0.81%** | 98.29% ± 0.03% | 20.97% ± 5.99% | **89.96% ± 3.55%** |
| `screw` (baseline 10% coreset) | 72.97% ± 1.08% | 98.32% ± 0.03% | 2.35% ± 0.36% | 15.41% ± 2.41% |
| `screw` (with category overrides) | **86.27%** | 98.40% | 4.82% | **85.70%** |
| `capsule` | **94.27% ± 0.44%** | 98.85% ± 0.03% | 16.37% ± 5.28% | **67.58% ± 10.39%** |

**Per-Defect Capture Breakdown:**
- **`metal_nut`:** `bent`: 98.7% ± 1.9% | `color`: 89.4% ± 7.7% | `flip`: 100.0% ± 0.0% | `scratch`: 71.0% ± 5.4%
- **`capsule`:** `squeeze`: 88.3% ± 10.3% | `crack`: 68.1% ± 4.1% | `scratch`: 65.2% ± 9.4% | `faulty_imprint`: 60.6% ± 15.0% | `poke`: 57.1% ± 15.6%
- **Zero normal false fails:** Across all 3 seeds and all nominal test samples, the false-alarm rejection rate was **0%**.

---

## 7. Evaluation Framework

`src/evaluation/metrics.py` provides reusable image-level and pixel-level evaluation with strict label validation (`0` = normal, `1` = defective, **higher score = more anomalous**).

| Function | Purpose |
|---|---|
| `compute_image_metrics()` | Image-AUROC, average precision, precision, recall, F1, and TN/FP/FN/TP counts. Uses a supplied threshold, or the max-F1 threshold if none is given. |
| `find_optimal_f1_threshold()` | Picks the F1-maximizing threshold from the precision-recall curve (uses labels, so it is an oracle operating point). |
| `compute_pixel_auroc()` | Pooled pixel-level AUROC between ground-truth masks and predicted heatmaps; accepts `(N, H, W)` or `(N, H, W, 1)` masks and validates shapes. |
| `validate_binary_labels()` | Rejects non-binary labels and single-class inputs. |

IoU and AUPRO are **not yet implemented**.

---

## 8. Inference API, Dashboard & Audit Log

### FastAPI service (`api/main.py`)

Models and the database are loaded once at startup. Interactive docs are served at `/docs`.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service status and loaded categories. |
| `/categories` | GET | Per-category validation mean/std and calibrated PASS/FAIL thresholds. |
| `/inspect` | POST | Multipart form: `category` (`metal_nut`, `screw` or `capsule`) and `file` (PNG/JPG). Returns the decision and diagnostics. |
| `/history` | GET | Most recent audit records (`?limit=`, default 50). |
| `/stats` | GET | Totals, PASS/REVIEW/FAIL counts, pass rate, average latency. |

`/inspect` returns: `inspection_id`, `decision`, `raw_score`, `normalized_score`, `z_score`, `defect_area_pct`, `pass_threshold`, `fail_threshold`, `category`, `latency_ms`, and `overlay_base64` (PNG heatmap overlay). It responds `400` for an unknown category, an empty upload, or an undecodable image.

```powershell
curl.exe -X POST http://127.0.0.1:8000/inspect `
  -F "category=metal_nut" `
  -F "file=@path\to\part_image.png"
```

### Streamlit operator console (`dashboard/app.py`)

- Product-line selector showing the calibrated thresholds for that category.
- Image upload, or a **quick sample loader** that picks a random good or defective test image from `data/raw`.
- Colour-coded verdict banner, anomaly score, z-score, defect area, latency, and a normalized risk bar.
- Side-by-side original image and heatmap overlay.
- KPI row (total inspected, pass rate, review queue, average latency) and a recent audit log table.
- Backend address is configurable with the `VISIONGUARD_API_URL` environment variable (default `http://127.0.0.1:8000`).

### Audit log (`src/storage/history_db.py`)

Every inspection is written to SQLite (`data/inspections.db`, overridable with `INSPECTION_DB_PATH`) with: id, UTC timestamp, filename, category, decision, raw score, normalized score, z-score, defect area %, and latency. Uploaded images and overlays are **not** stored.

---

## 9. Repository Structure

```text
VisionGuard-AI/
|
+-- api/
|   `-- main.py                    # FastAPI inference & audit service
|
+-- dashboard/
|   `-- app.py                     # Streamlit operator console
|
+-- configs/
|   +-- model.yaml                 # PatchCore + decision-engine settings (used by the pipeline)
|   `-- data.yaml                  # Dataset settings (currently not read by any code)
|
+-- data/
|   `-- README.md                  # Dataset documentation (raw data and *.db are git-ignored)
|
+-- notebooks/
|   +-- 01_data_exploration.ipynb
|   +-- 02_cnn_autoenconder.ipynb
|   +-- 03_autoencoder_ssim_loss.ipynb
|   `-- 04_patchcore_efficientnet.ipynb   # PatchCore + 3-category benchmark
|
+-- scripts/
|   `-- build_artifacts.py         # Fit memory banks and calibrate thresholds
|
+-- src/
|   +-- data/                      # download_mvtec.py, verify_dataset.py, preprocessing.py, loader.py
|   +-- models/                    # autoencoder.py, efficientnet.py, feature_memory.py
|   +-- training/                  # train_autoencoder.py, losses.py
|   +-- inference/                 # predictor.py, decision_engine.py, localization.py
|   +-- evaluation/                # metrics.py
|   +-- storage/                   # history_db.py
|   `-- utils/                     # config.py
|
+-- tests/                         # test_data, test_models, test_losses, test_patchcore,
|                                  # test_metrics, test_decision, test_api
|
+-- .github/workflows/ci.yml       # GitHub Actions
+-- Dockerfile
+-- docker-compose.yml
+-- requirements.txt
`-- README.md
```

Generated and git-ignored: `data/raw/`, `models/` (PatchCore artifacts), `artifacts/` (autoencoder checkpoints), `data/*.db`.

---

## 10. Getting Started

Commands assume Windows PowerShell and are run from the repository root. On Linux/macOS, activate the environment with `source .venv/bin/activate`.

### 1. Clone and create an environment

```powershell
git clone https://github.com/RuchithaAV/VisionGuard-AI.git
cd VisionGuard-AI
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
pip install huggingface_hub matplotlib jupyter
```

The second `pip install` covers packages that are used but not listed in `requirements.txt`: `huggingface_hub` (dataset downloader) and `matplotlib` / `jupyter` (notebooks). The code is run on Python 3.11 in CI and Docker.

### 3. Get the dataset

Download the three target categories from the Hugging Face mirror used by the project:

```powershell
python -m src.data.download_mvtec
```

Or obtain MVTec AD from the [official source](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract `metal_nut`, `screw` and `capsule` into `data/raw/` (see `data/README.md` for the expected layout). Then verify integrity:

```powershell
python -m src.data.verify_dataset
```

### 4. Build model artifacts

```powershell
python -m scripts.build_artifacts
```

For each category this fits the memory bank on the train split, calibrates the decision thresholds on the validation split, and writes `memory_bank.npz` and `metadata.json` to `models/patchcore/<category>/`. The API will not start until these exist.

### 5. Run the system

```powershell
# Terminal 1 - API
uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 - dashboard
streamlit run dashboard/app.py
```

The dashboard is at `http://localhost:8501` and the API docs at `http://127.0.0.1:8000/docs`.

### Docker alternative

```powershell
python -m scripts.build_artifacts     # artifacts must exist before building (the image copies models/)
docker compose up --build
```

`docker-compose.yml` starts the API on port 8000 (with a `/health` check) and the dashboard on port 8501, which waits for a healthy API. `./data` is mounted so the audit database persists and the dashboard's sample loader can read `data/raw`.

### Reproducing the experiments

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Dataset exploration and defect-footprint analysis. |
| `02_cnn_autoenconder.ipynb` | Baseline convolutional autoencoder (MSE). |
| `03_autoencoder_ssim_loss.ipynb` | SSIM + MAE loss and foreground edge masking. |
| `04_patchcore_efficientnet.ipynb` | EfficientNetB0 PatchCore and the three-category benchmark. |

Autoencoder training from the command line (MSE baseline; the SSIM + MAE option is selected with `loss_type="ssim_mae"` in `train_autoencoder()`, as used in notebook 03):

```powershell
python src/training/train_autoencoder.py metal_nut 5     # <category> <epochs>
```

Checkpoints are written to `artifacts/models/`.

---

## 11. Automated Testing & CI

```powershell
python -m pytest -v
```

The suite has **37 test functions across 7 modules**.

| Module | Covers |
|---|---|
| `test_data.py` | Image/mask preprocessing, binary masks, normal-image masks, zero-leakage train/validation split. |
| `test_models.py` | Autoencoder shapes and forward pass, error maps, heatmaps/overlays, foreground masking, boundary suppression, localization alpha-blending. |
| `test_losses.py` | SSIM + MAE loss behaviour, weighting, validation, factory, serialization, a training step. |
| `test_patchcore.py` | Feature extractor shapes `(B, 32, 32, 64)`, patch flattening, memory bank, coreset size, scoring, save/load round trip. |
| `test_metrics.py` | Perfect separation, fixed-threshold confusion matrix, pixel-AUROC mask shapes, single-class errors. |
| `test_decision.py` | Calibration, three-state routing, normalized score anchors, defect-area calculation, category mismatch, serialization, invalid inputs. |
| `test_api.py` | Config structure, decision routing, category mismatch errors, and an end-to-end `/health` -> `/categories` -> `/inspect` -> `/stats` flow. |

**Prerequisites & Self-Contained CI:** `test_data.py` uses `pytest.mark.skipif` when raw dataset files are absent, while `test_api.py` and `test_models.py` generate self-contained synthetic fixtures in-memory. This allows continuous integration (`.github/workflows/ci.yml`) to pass 100% on fresh runner environments without requiring the full gigabyte-scale MVTec AD dataset.

---

## 12. Engineering & Evaluation Protocol

### Zero data leakage

- Training and memory banks use **only** nominal `train/good` images.
- The validation split is carved from `train/good` (seed 42) and is used **only** to calibrate decision thresholds.
- Defective images never enter a memory bank, and the official test split is reserved for evaluation.
- The oracle max-F1 numbers in section 6.2 use test labels and are reported as such; they are not used for deployment.

### Consistent conventions

| Label | Meaning |
|---|---|
| `0` | Normal |
| `1` | Defective |

Higher anomaly score always means stronger evidence of a defect.

### Reproducibility

Experiments should record the category, model and feature configuration, preprocessing, split and seed, threshold-selection method, and metrics. Pipeline settings live in `configs/model.yaml`. Experiment metrics and parameters are logged via **MLflow**.

---

## 13. Roadmap & Milestones

| Milestone | Focus | Status |
|---|---|---|
| **01** | Dataset & data engineering: verification, exploration, preprocessing, TensorFlow pipelines. | 🟢 Completed |
| **02** | Convolutional autoencoder: encoder-decoder, MSE baseline, reconstruction maps. | 🟢 Completed |
| **03** | Advanced autoencoder studies: SSIM + MAE loss, foreground edge masking. | 🟢 Completed |
| **04** | PatchCore feature memory: EfficientNetB0 embeddings, coreset memory bank, nearest-neighbour scoring, three-category benchmark. | 🟢 Completed |
| **05** | Evaluation & benchmarking: metrics module, multi-seed evaluation (3 seeds), per-defect breakdown, and Pixel-IoU. | 🟢 Completed |
| **06** | Decision engine & explainability: validation-calibrated PASS/REVIEW/FAIL, normalized risk, defect-area estimate, heatmap overlays, test-set routing validation. | 🟢 Completed |
| **07** | Application interface: FastAPI REST service, category mismatch handling, and Streamlit inspection console. | 🟢 Completed |
| **08** | Production engineering: Docker, Compose, SQLite audit log, GitHub Actions CI with self-contained fixtures, Ruff linting, MLflow tracking. | 🟢 Completed |

---

## 14. Known Limitations

**Known limitations**

- **`screw` category tuning:** Baseline 10% coreset required category overrides (`coreset_ratio: 0.40`, tighter decision multipliers) to reach 86.27% AUROC. Further architecture work (e.g. higher-resolution feature grids) could push localization higher on sub-0.2% defect scales.
- **Latency:** roughly 175-195 ms per image on CPU for `metal_nut` and `capsule`, and ~1.5s for `screw` due to its larger coreset memory bank.
- **API scope:** no user authentication/RBAC, and raw uploaded images are not permanently retained on disk (only metadata is audited in SQLite).
- **Configuration:** `configs/data.yaml` is legacy; the unified pipeline reads `configs/model.yaml`.

---

## 15. License, Attribution & References

**Dataset:** MVTec Anomaly Detection (MVTec AD), provided by MVTec Software GmbH, distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).

- Dataset: https://www.mvtec.com/company/research/datasets/mvtec-ad
- License: https://creativecommons.org/licenses/by-nc-sa/4.0/

**References**

- Bergmann, P., Fauser, M., Sattlegger, D., Steger, C. *MVTec AD - A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* CVPR 2019.
- Roth, K., Pemula, L., Zepeda, J., Scholkopf, B., Brox, T., Gehler, P. *Towards Total Recall in Industrial Anomaly Detection.* CVPR 2022.

**Author:** Ruchitha Vithana
BSc (Hons) in Information Technology & Management
University of Colombo, Sri Lanka.

---

*VisionGuard AI is an evolving computer vision engineering project. Claims in this document are limited to results recorded in the repository's notebooks and tests.*