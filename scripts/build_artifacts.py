"""
VisionGuard AI - Production Artifact Builder

Fits PatchCoreMemoryBank on nominal training images and calibrates
DecisionEngine thresholds strictly on held-out normal validation splits
(zero test leakage) for each configured MVTec AD category.
"""

import time
from pathlib import Path
import numpy as np

from src.data.loader import create_autoencoder_datasets
from src.models.efficientnet import MultiScaleFeatureExtractor
from src.models.feature_memory import PatchCoreMemoryBank
from src.inference.decision_engine import DecisionEngine
from src.utils.config import load_config


def build_all_artifacts(config_path: str = "configs/model.yaml") -> None:
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    pc_cfg = cfg["patchcore"]
    dec_cfg = cfg["decision_engine"]

    artifact_root = Path(pc_cfg["artifact_root"])
    artifact_root.mkdir(parents=True, exist_ok=True)

    # Instantiate frozen EfficientNetB0 backbone ONCE and share across categories
    extractor = MultiScaleFeatureExtractor(
        input_shape=tuple(pc_cfg["input_shape"]),
        layer_names=pc_cfg["layer_names"],
        pool_size=pc_cfg["pool_size"],
    )

    print(f"Building VisionGuard AI artifacts into '{artifact_root}'...\n")

    for category in data_cfg["categories"]:
        t0 = time.perf_counter()
        train_ds, val_ds, n_train, n_val = create_autoencoder_datasets(
            category=category,
            data_dir=data_cfg["raw_dir"],
            batch_size=data_cfg["batch_size"],
            val_split=data_cfg["val_split"],
            random_seed=data_cfg["random_seed"],
        )

        # 1. Fit PatchCore memory bank on nominal train split
        bank = PatchCoreMemoryBank(
            feature_extractor=extractor,
            coreset_ratio=pc_cfg["coreset_ratio"],
            n_neighbors=pc_cfg["n_neighbors"],
        ).fit(train_ds)

        # 2. Score held-out normal validation split (zero test leakage)
        val_scores = []
        for v_batch, _ in val_ds:
            for v_img in v_batch.numpy():
                score, _ = bank.score_image(
                    v_img,
                    gaussian_kernel=pc_cfg["gaussian_kernel"],
                    sigma=pc_cfg["gaussian_sigma"],
                )
                val_scores.append(score)
        val_scores = np.array(val_scores, dtype=np.float64)

        # 3. Calibrate 3-state DecisionEngine
        engine = DecisionEngine(
            pass_std_multiplier=dec_cfg["pass_std_multiplier"],
            fail_std_multiplier=dec_cfg["fail_std_multiplier"],
            category=category,
        ).fit(val_scores)

        # 4. Save memory bank + embedded calibration metadata
        out_dir = artifact_root / category
        bank.save(
            str(out_dir),
            metadata={
                "category": category,
                "n_train": n_train,
                "n_val": n_val,
                "decision_engine": engine.to_dict(),
            },
        )

        elapsed = time.perf_counter() - t0
        print(
            f"[OK] {category:<10} | train={n_train}, val={n_val} | "
            f"val={engine.val_mean:.2f}+-{engine.val_std:.2f} | "
            f"PASS<={engine.pass_threshold:.2f} | FAIL>{engine.fail_threshold:.2f} | "
            f"{elapsed:.2f}s"
        )


if __name__ == "__main__":
    build_all_artifacts()
