# VisionGuard AI - Industrial Visual Quality Inspection & Defect Detection System

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Framework: TensorFlow/Keras](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-orange.svg)](https://tensorflow.org/)
[![Dataset: MVTec AD](https://img.shields.io/badge/Dataset-MVTec%20AD-green.svg)](https://www.mvtec.com/company/research/datasets/mvtec-ad)
[![Status](https://img.shields.io/badge/Status-Research%20%26%20Development-yellow.svg)]()

> **VisionGuard AI** is an industry-oriented computer vision research project designed for automated, unsupervised visual quality inspection in manufacturing environments. It explores deep learning-based anomaly detection to identify defective products, localize anomalous regions through spatial heatmaps, and evaluate inspection performance using image-level and pixel-level metrics.

---

## 1. Problem Statement

In modern smart manufacturing, visual quality inspection is essential for maintaining product quality and minimizing defective output. However, conventional inspection approaches face several challenges:

- **Human Inspection Limitations:** Manual inspection is vulnerable to fatigue, subjective judgment, operational costs, and scalability limitations.
- **Limited Defective Samples:** Industrial defects are often rare, diverse, and difficult to predict. Traditional supervised classification requires sufficient labeled examples of different defect types.
- **Microscopic Defect Footprints:** Small scratches, cracks, and surface irregularities can occupy less than 0.25% of the product image. Global image-level representations may overlook these localized abnormalities.
- **Unseen Defects:** Manufacturing environments can introduce previously unseen defect patterns that are not represented in training datasets.

VisionGuard AI investigates **unsupervised anomaly detection**, where models learn normal product characteristics primarily from defect-free training images and identify deviations during inspection.

The project explores two complementary approaches:

1. Reconstruction-based anomaly detection using a Convolutional Autoencoder.
2. Feature-based anomaly detection using an EfficientNetB0 feature extractor and PatchCore-inspired memory bank.

The objective is to investigate which approach provides more effective anomaly detection and localization across different industrial product categories.

---

## 2. Technical Architecture & Multi-Model Approach

VisionGuard AI follows a multi-model anomaly detection architecture.

```text
                    Incoming Industrial Image
                              |
                   Image Preprocessing
                    (256 x 256 x 3)
                              |
              +---------------+---------------+
              |                               |
              v                               v
    +----------------------+       +--------------------------+
    |       MODEL 1        |       |         MODEL 2          |
    |                      |       |                          |
    | Convolutional        |       | PatchCore-Inspired       |
    | Autoencoder          |       | Feature Memory           |
    +----------+-----------+       +------------+-------------+
               |                               |
               v                               v
       Reconstruction Error          EfficientNetB0 Features
               |                               |
               v                               v
       Reconstruction Map             Multi-scale Embeddings
                                               |
                                               v
                                      Nominal Feature Memory
                                               |
                                               v
                                      Nearest Neighbor Search
                                               |
                                               v
                                       Patch Anomaly Scores
               |                               |
               +---------------+---------------+
                               |
                               v
                    Anomaly Evaluation
                               |
                 +-------------+-------------+
                 |                           |
                 v                           v
          Image-level Metrics          Pixel-level Metrics
          Image-AUROC                  Pixel-AUROC
          Average Precision            Anomaly Heatmaps
          Precision / Recall
          F1-score
          Confusion Matrix
```

### Model 1: Custom Convolutional Autoencoder

**Architecture:** A convolutional encoder-decoder developed using the Keras Functional API.

- **Learning Principle:** Learns compressed representations of nominal product images and reconstructs their original appearance.
- **Input:** Standardized RGB images of shape `(256, 256, 3)`.
- **Latent Representation:** The encoder compresses image information into a lower-dimensional feature representation.
- **Baseline Loss:** Mean Squared Error (MSE).
- **Advanced Loss:** Custom `SSIMMAELoss`, combining Structural Similarity and Mean Absolute Error.

The combined loss is defined as:

\[
\mathcal{L} =
\alpha(1-\operatorname{SSIM}(X,\hat X))
+(1-\alpha)\operatorname{MAE}(X,\hat X)
\]

Where:

- \(X\) represents the original image.
- \(\hat X\) represents the reconstructed image.
- \(\alpha\) controls the contribution of structural similarity.

**Localization Enhancements:**

- Adaptive foreground segmentation.
- Otsu thresholding.
- Morphological contour processing.
- Elliptical boundary erosion.
- Reconstruction error map generation.
- Gaussian smoothing and heatmap visualization.

These techniques investigate the reduction of reconstruction artefacts caused by object boundaries and background variations.

### Model 2: PatchCore-Inspired Feature Memory Bank

The second approach introduces pretrained deep feature representations for local anomaly detection.

Unlike reconstruction-based approaches, this method compares local feature representations against a memory bank containing normal product patterns.

**Backbone: EfficientNetB0**

- Uses an ImageNet-pretrained EfficientNetB0.
- Extracts intermediate feature representations from `block2b_add` and `block3b_add`.
- Freezes the pretrained backbone weights.
- Aligns multi-scale feature maps into a common spatial resolution.
- Applies locally aware average pooling.
- Concatenates the extracted representations into unified patch embeddings.

**Feature Representation:**

For a 256 × 256 input image:

| Property | Output |
|---|---|
| Input image | 256 × 256 × 3 |
| Feature grid | 32 × 32 |
| Feature dimensions | 64 |
| Total patches per image | 1,024 |
| Flattened representation | 1,024 × 64 |

The resulting feature tensor has the shape:

`(Batch Size, 32, 32, 64)`

**Feature Memory Construction**

The `PatchCoreMemoryBank` class implements the nominal feature-memory mechanism.

1. Extract patch embeddings from nominal training images.
2. Collect the embeddings into a feature representation pool.
3. Apply configurable coreset subsampling.
4. Construct a reduced memory bank.
5. Fit a Euclidean nearest-neighbor search structure.

The default coreset retention ratio is 5%.

The current implementation uses deterministic random sampling with a fixed seed (`42`) to select representative memory vectors.

**Anomaly Scoring**

During inference:

1. Extract feature embeddings from the test image.
2. Flatten the spatial feature grid into patch vectors.
3. Calculate nearest-neighbor distances against the nominal memory bank.
4. Reshape the patch distances into a spatial anomaly map.
5. Calculate the image-level anomaly score using the maximum patch distance.
6. Upscale the anomaly map to the original image resolution.
7. Apply Gaussian smoothing.

The output consists of:

- Image-level anomaly score.
- Spatial patch anomaly distances.
- Full-resolution anomaly heatmap.

**Important:** This is a PatchCore-inspired implementation. Its current random coreset sampling strategy should not be confused with the greedy k-center selection used in the original PatchCore methodology.

---

## 3. Dataset & Empirical Discoveries

VisionGuard AI utilizes the **MVTec Anomaly Detection (MVTec AD)** benchmark dataset.

The current experiments focus on three structurally different industrial product categories.

| Category | Structural Profile | Raw Resolution | Native Mode | Defect Types | Defect Footprint |
|---|---|---|---|---|---|
| `metal_nut` | Rigid symmetric metal ring with threading | 700 × 700 | RGB | bent, color, flip, scratch | 0.436% – 48.439% |
| `screw` | Elongated threaded mechanical fastener | 1024 × 1024 | Grayscale | manipulated_front, scratch_head, scratch_neck, thread_side, thread_top | 0.151% – 0.649% |
| `capsule` | Non-metallic pharmaceutical pill | 1000 × 1000 | RGB | crack, faulty_imprint, poke, scratch, squeeze | 0.249% – 7.694% |

### Verified Exploratory Findings

**1. Grayscale vs. Color**

The `screw` category contains native single-channel grayscale images, whereas `metal_nut` and `capsule` contain RGB images.

The preprocessing pipeline standardizes the inputs to three channels to support the pretrained EfficientNetB0 architecture.

**2. Microscopic Defect Scales**

Some industrial defects occupy an extremely small proportion of the image.

Examples:

- `screw/thread_side`: approximately 0.151% of the image.
- `capsule/crack`: approximately 0.249% of the image.

This highlights the importance of localized feature extraction instead of relying exclusively on global image representations.

**3. Interpolation Artifacts**

Exploratory preprocessing experiments identified the importance of interpolation methods when resizing ground-truth segmentation masks.

- Bilinear interpolation can introduce intermediate pixel values.
- Nearest-neighbor interpolation preserves binary mask labels.

Therefore, nearest-neighbor interpolation is used for binary mask resizing.

---

## 4. Experimental Results & Autoencoder Progression

Before introducing the feature-memory architecture, several experiments were conducted to investigate reconstruction-based anomaly detection.

The experiments were evaluated using the 115-image `metal_nut` test set, consisting of 22 nominal images and 93 defective images across four defect classes.

### Autoencoder Experimental Comparison

| Stage | Architecture | Loss Function | Post-processing | Normal Mean Score | Defect Mean Score | Image-AUROC |
|---|---|---|---|---|---|---|
| 01 | Baseline Conv Autoencoder | MSE | Raw error map | 0.2647 ± 0.0306 | 0.2419 ± 0.0246 | 25.61% |
| 02 | Structural Loss Autoencoder | SSIM + MAE | Raw error map | 0.2820 ± 0.0243 | 0.2675 ± 0.0153 | 30.21% |
| 03 | Foreground Edge-Masked AE | SSIM + MAE | 8 px morphological edge erosion | 0.1837 ± 0.0331 | 0.1637 ± 0.0310 | 31.67% |

### Architectural Diagnostics

**Stage 01: Baseline MSE Autoencoder**

The initial experiment revealed that reconstruction errors were strongly influenced by object boundaries and rotational variations.

For rigid metallic objects, small differences in alignment and geometry generated large reconstruction errors around the product silhouette.

Consequently, some nominal samples received higher anomaly scores than defective samples, resulting in poor anomaly separation.

**Stage 02: Structural Similarity Loss**

The custom `SSIMMAELoss` was introduced to incorporate structural information alongside pixel-level reconstruction differences.

The recorded Image-AUROC increased from 25.61% to 30.21%.

**Stage 03: Foreground Edge Masking**

Foreground segmentation and morphological erosion were introduced to suppress boundary-related reconstruction artefacts.

This experiment achieved an Image-AUROC of 31.67%.

### Key Research Findings

- Reconstruction-based anomaly detection can be sensitive to rotational misalignment and object boundaries.
- Structural similarity provides an additional learning objective beyond pixel-wise reconstruction.
- Foreground masking can reduce irrelevant boundary errors.
- Local deep feature representations offer an alternative direction for identifying subtle defects.

The limitations observed in the autoencoder experiments motivated the transition towards PatchCore-inspired feature-memory anomaly detection.

**Note:** These results represent the specific `metal_nut` experiment. They should not be interpreted as general performance across the entire MVTec AD dataset.

### PatchCore Development Progress

The following components have now been implemented:

- Multi-scale EfficientNetB0 feature extraction.
- Feature-map alignment and patch embedding generation.
- Feature-memory construction.
- Configurable coreset subsampling.
- Euclidean nearest-neighbor search.
- Image-level anomaly scoring.
- Spatial anomaly heatmap generation.
- PatchCore component unit tests.

A complete benchmark comparing both architectures across the selected product categories remains part of the ongoing evaluation work.

No comparative PatchCore performance score is claimed until the corresponding experiments have been completed and validated.

---

## 5. Evaluation Framework

The evaluation module is implemented in:

`src/evaluation/metrics.py`

It provides reusable functions for evaluating image-level and pixel-level anomaly detection.

### Image-Level Evaluation

Implemented functionality:

| Metric | Description |
|---|---|
| Image-AUROC | Measures the ability to distinguish normal and defective images across different score thresholds. |
| Average Precision | Evaluates precision-recall performance. |
| Precision | Measures the proportion of predicted defects that are actual defects. |
| Recall | Measures the proportion of actual defects successfully identified. |
| F1-score | Harmonic mean of precision and recall. |
| Confusion Matrix | Provides TN, FP, FN and TP counts. |

**Automatic Threshold Optimization**

The evaluation module includes `find_optimal_f1_threshold()`.

It evaluates candidate thresholds using the precision-recall curve and selects the threshold that maximizes F1-score.

The `compute_image_metrics()` function also supports evaluation using a manually supplied threshold.

### Pixel-Level Evaluation

The function `compute_pixel_auroc()` evaluates anomaly localization using:

- Ground-truth defect segmentation masks.
- Predicted continuous anomaly heatmaps.
- Pixel-level binary labels.
- Spatial shape validation.

The implementation supports masks with shapes:

- `(N, H, W)`
- `(N, H, W, 1)`

The predicted anomaly maps must have matching spatial dimensions.

### Evaluation Reliability

The evaluation module also includes:

- Binary label validation.
- Validation of normal and defective classes.
- Input shape consistency checks.
- Explicit anomaly score direction (`higher score = more anomalous`).

**Additional metrics such as IoU and AUPRO are future extensions and are not currently implemented in the evaluation module.**

---

## 6. Repository Structure

```text
VisionGuard-AI/
│
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   └── training.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_cnn_autoenconder.ipynb
│   ├── 03_autoencoder_ssim_loss.ipynb
│   ├── 04_patchcore_efficientnet.ipynb
│   └── 05_model_comparison.ipynb
│
├── src/
│   │
│   ├── data/
│   │   ├── verify_dataset.py
│   │   ├── preprocessing.py
│   │   └── loader.py
│   │
│   ├── models/
│   │   ├── autoencoder.py
│   │   ├── efficientnet.py
│   │   └── feature_memory.py
│   │
│   ├── training/
│   │   └── losses.py
│   │
│   ├── inference/
│   │   └── localization.py
│   │
│   └── evaluation/
│       └── metrics.py
│
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_losses.py
│   ├── test_patchcore.py
│   └── test_metrics.py
│
├── requirements.txt
└── README.md
```

*The structure above highlights the core project components and should be kept synchronized with the actual repository as additional modules are introduced.*

### Important Source Files

| File | Responsibility |
|---|---|
| `src/data/verify_dataset.py` | Dataset structure and integrity verification. |
| `src/data/preprocessing.py` | Image and mask preprocessing. |
| `src/data/loader.py` | TensorFlow dataset construction and data splitting. |
| `src/models/autoencoder.py` | Functional convolutional autoencoder. |
| `src/models/efficientnet.py` | Multi-scale EfficientNetB0 feature extraction. |
| `src/models/feature_memory.py` | PatchCore-inspired memory bank and anomaly scoring. |
| `src/training/losses.py` | Custom SSIM + MAE reconstruction loss. |
| `src/inference/localization.py` | Reconstruction error maps and heatmap visualization. |
| `src/evaluation/metrics.py` | Image-level and pixel-level evaluation metrics. |

---

## 7. Development Roadmap & Milestones

| Milestone | Focus | Key Deliverables | Status |
|---|---|---|---|
| **01** | Dataset & Data Engineering | Dataset verification, exploratory analysis, preprocessing and TensorFlow pipelines. | 🟢 Completed |
| **02** | Convolutional Autoencoder | Functional encoder-decoder, MSE baseline and reconstruction maps. | 🟢 Completed |
| **03** | Advanced Autoencoder Studies | SSIM + MAE loss, foreground edge masking and experimental ablation. | 🟢 Completed |
| **04** | PatchCore Feature Memory | EfficientNetB0 feature extraction, patch embeddings, feature memory, coreset sampling and nearest-neighbor scoring. | 🟢 Implementation completed; benchmarking ongoing |
| **05** | Evaluation & Benchmarking | Image-AUROC, Pixel-AUROC, Average Precision, threshold analysis and model comparison. | 🟡 Core metrics and unit tests implemented; full benchmarking ongoing |
| **06** | Decision Engine & Explainability | Defect footprint estimation, interpretable overlays and calibrated PASS / REVIEW / FAIL decisions. | ⚪ Planned |
| **07** | Application Interface | FastAPI REST API and Streamlit inspection dashboard. | ⚪ Planned |
| **08** | Production Engineering | Docker, integration testing, experiment tracking and CI/CD. | ⚪ Planned |
| **09** | Domain Generalization | External dataset evaluation, VisA experiments and technical reporting. | ⚪ Planned |

---

## 8. Getting Started

### 1. Clone the Repository

```powershell
git clone https://github.com/RuchithaAV/VisionGuard-AI.git

cd VisionGuard-AI
```

### 2. Create a Virtual Environment

```powershell
python -m venv .venv

.venv\Scripts\activate
```

### 3. Install Dependencies

```powershell
pip install --upgrade pip

pip install -r requirements.txt
```

### 4. Dataset Setup

Download the MVTec AD dataset from the official source:

https://www.mvtec.com/company/research/datasets/mvtec-ad

Place the required categories in the project's expected dataset directory structure.

The current experiments focus on:

- `metal_nut`
- `screw`
- `capsule`

### 5. Verify Dataset Integrity

```powershell
python src/data/verify_dataset.py
```

The verification utility checks:

- Required category directories.
- Training and testing directory structures.
- Ground-truth mask availability.
- Correspondence between defect images and masks.
- Dataset image and defect-class summaries.

### 6. Run the Notebooks

Open the project using Jupyter Notebook, VS Code, or Antigravity.

Select the `.venv` Python environment as the notebook kernel.

The main experimental notebooks are:

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Dataset exploration and defect-footprint analysis. |
| `02_cnn_autoenconder.ipynb` | Baseline convolutional autoencoder experiments. |
| `03_autoencoder_ssim_loss.ipynb` | Structural loss and foreground edge-masking experiments. |
| `04_patchcore_efficientnet.ipynb` | EfficientNetB0 feature extraction and feature-memory experiments. |
| `05_model_comparison.ipynb` | Evaluation and model-comparison experiments. |

---

## 9. Automated Testing

VisionGuard AI uses **pytest** for component-level validation.

Execute the complete test suite using:

```powershell
python -m pytest -v
```

### Test Modules

**1. `tests/test_data.py`**

Validates the data pipeline:

- Image preprocessing dimensions and value ranges.
- Binary mask constraints.
- Normal-image mask handling.
- Training and validation split integrity.

**2. `tests/test_models.py`**

Validates model architecture and localization:

- Encoder and decoder output shapes.
- Autoencoder forward-pass behavior.
- Reconstruction error maps.
- Heatmap generation and visualization.
- Foreground mask extraction.
- Boundary suppression and anomaly scoring.

**3. `tests/test_losses.py`**

Validates custom reconstruction losses:

- Zero loss for identical images.
- Structural damage sensitivity.
- SSIM and MAE weighting.
- Invalid parameter handling.
- Loss factory functionality.
- Keras serialization.
- Training execution.

**4. `tests/test_patchcore.py`**

Validates the feature-memory architecture:

- EfficientNetB0 multi-scale feature extraction.
- Expected `(B, 32, 32, 64)` output dimensions.
- Patch-vector flattening.
- Memory bank construction.
- Coreset subsampling.
- Nearest-neighbor anomaly scoring.
- Heatmap output dimensions.

**5. `tests/test_metrics.py`**

Validates the evaluation framework:

- Perfect image-level anomaly separation.
- Fixed-threshold confusion matrix calculations.
- Pixel-AUROC calculations.
- Support for 3D and 4D ground-truth masks.
- Invalid single-class label handling.

Run the tests in your local environment to obtain the current test count and execution results.

---

## 10. Engineering & Evaluation Protocol

The project follows several principles to improve the reliability of its experiments.

### Zero Data Leakage

- Training uses nominal (`train/good`) samples.
- Validation data is separated from the nominal training pool.
- Defective test images are not used to construct the feature memory bank.
- The official test dataset is reserved for evaluation.

### Consistent Evaluation

All evaluation functions follow these label conventions:

| Label | Meaning |
|---|---|
| `0` | Normal |
| `1` | Defective |

Higher anomaly scores indicate stronger evidence of a defect.

### Reproducibility

Experiments should document:

- Product category.
- Model architecture.
- Feature extraction configuration.
- Preprocessing methods.
- Training and validation settings.
- Threshold selection methodology.
- Evaluation metrics.

### Performance Analysis

Future benchmarking will investigate:

- Image-level anomaly discrimination.
- Pixel-level defect localization.
- Precision-recall trade-offs.
- False-positive and false-negative behavior.
- Inference latency.
- Memory-bank size and computational requirements.
- Cross-category generalization.

---

## 11. Future Improvements

The next development stages will focus on transforming the research implementation into a more complete industrial inspection system.

### Model Improvements

- Investigate greedy coreset selection and alternative memory reduction strategies.
- Evaluate different EfficientNet feature extraction layers.
- Investigate feature normalization and anomaly score calibration.
- Improve small-defect localization.
- Compare reconstruction-based and feature-memory-based detection.
- Evaluate additional industrial product categories.

### Evaluation Improvements

- Intersection over Union (IoU).
- Area Under the Per-Region Overlap (AUPRO).
- Per-category benchmarking.
- Defect-specific performance analysis.
- Inference latency and memory profiling.

### Explainability

- Defect footprint estimation.
- Improved anomaly heatmap visualization.
- Inspection overlays.
- Interpretable PASS / REVIEW / FAIL decision thresholds.

### Deployment

- FastAPI inference endpoints.
- Streamlit inspection dashboard.
- Docker-based deployment.
- Experiment tracking with MLflow.
- Automated testing and CI/CD integration.

### Domain Generalization

- Evaluation using the VisA industrial anomaly detection dataset.
- Investigation of unseen product categories.
- Cross-dataset robustness analysis.

These are planned improvements and are not represented as completed capabilities.

---

## 12. License & Attribution

**Dataset:** MVTec Anomaly Detection (MVTec AD), provided by MVTec Software GmbH.

Dataset reference:

https://www.mvtec.com/company/research/datasets/mvtec-ad

The MVTec AD dataset is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).

https://creativecommons.org/licenses/by-nc-sa/4.0/

**Author:** Ruchitha Vithana

BSc (Hons) in Information Technology & Management  
University of Colombo, Sri Lanka.

---

*VisionGuard AI is an evolving computer vision research and engineering project. Its architecture, experiments and evaluation framework are being developed incrementally, with an emphasis on reproducibility, reliable anomaly detection and practical industrial applicability.*