# VisionGuard AI - Dataset Documentation

## Primary Dataset: MVTec Anomaly Detection (MVTec AD)

- **Official Website:** [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). Strictly for non-commercial research/educational use.
- **Data Policy:** Do NOT commit raw dataset files (`.png`, `.tar.xz`, `.zip`) to the GitHub repository. Keep raw files in `data/raw/` (ignored by `.gitignore`).

---

## Initial Target Categories

We focus on three industrial categories exhibiting distinct structural and surface characteristics:

1. **`metal_nut`**: Rigid metallic circular surface; defects include `bent`, `color`, `flip`, and `scratch`.
2. **`screw`**: Elongated threaded fastener with orientation and thread variance; defects include `manipulated_front`, `scratch_head`, `scratch_neck`, `thread_side`, `thread_top`.
3. **`capsule`**: Non-metallic, colored pharmaceutical object with printed markings; defects include `crack`, `faulty_imprint`, `poke`, `scratch`, `squeeze`.

---

## Expected Folder Structure

Extract the categories into `data/raw/` so the tree appears as:

```text
data/
└── raw/
    ├── metal_nut/
    │   ├── train/
    │   │   └── good/              # Only normal images used for training
    │   ├── test/
    │   │   ├── good/              # Normal test images
    │   │   ├── bent/              # Defect subcategories
    │   │   ├── color/
    │   │   ├── flip/
    │   │   └── scratch/
    │   └── ground_truth/
    │       ├── bent/              # Pixel-level binary masks (_mask.png)
    │       ├── color/
    │       ├── flip/
    │       └── scratch/
    ├── screw/
    │   ├── train/good/
    │   ├── test/
    │   └── ground_truth/
    └── capsule/
        ├── train/good/
        ├── test/
        └── ground_truth/
```

---

## Acquisition Methods

### Option 1: Official MVTec Portal (Recommended)
1. Visit [MVTec AD Portal](https://www.mvtec.com/company/research/datasets/mvtec-ad).
2. Fill out the brief form (Name, Organization, Email, Academic/Non-commercial).
3. Access the direct archive links.

### Option 2: Kaggle CLI
If you have Kaggle CLI configured:
```bash
kaggle datasets download -d ipythonx/mvtec-ad -p data/raw/
```
Unzip and extract `metal_nut`, `screw`, and `capsule` into `data/raw/`.
