"""
VisionGuard AI - Reproducible Multi-Seed & Per-Defect Benchmark
Runs PatchCore + validation-calibrated DecisionEngine across multiple seeds,
reporting Image-AUROC, Pixel-AUROC, Pixel-IoU, routing rates, and per-defect recall.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data.loader import create_autoencoder_datasets
from src.data.preprocessing import load_and_preprocess_image, load_and_preprocess_mask
from src.evaluation.metrics import compute_image_metrics, compute_pixel_auroc, compute_pixel_iou
from src.inference.decision_engine import DecisionEngine
from src.models.efficientnet import MultiScaleFeatureExtractor
from src.models.feature_memory import PatchCoreMemoryBank
from src.utils.config import load_config


def run_benchmark(seeds: tuple[int, ...] = (42, 123, 999)) -> dict:
    cfg = load_config("configs/model.yaml")
    data_cfg = cfg["data"]
    pc_cfg = cfg["patchcore"]
    dec_cfg = cfg["decision_engine"]`n    overrides_cfg = cfg.get("category_overrides", {})
    img_size = tuple(data_cfg["image_size"])

    extractor = MultiScaleFeatureExtractor(
        input_shape=tuple(pc_cfg["input_shape"]),
        layer_names=pc_cfg["layer_names"],
        pool_size=pc_cfg["pool_size"],
    )

    report: dict = {"seeds": list(seeds), "categories": {}}

    for category in data_cfg["categories"]:`n        cat_ovr = overrides_cfg.get(category, {})`n        coreset_ratio = float(cat_ovr.get("coreset_ratio", pc_cfg["coreset_ratio"]))
        cat_img_size = tuple(cat_ovr.get("image_size", data_cfg["image_size"]))
        cat_in_shape = tuple(cat_ovr.get("input_shape", pc_cfg["input_shape"]))
        cat_layers = list(cat_ovr.get("layer_names", pc_cfg["layer_names"]))`n        pass_mult = float(cat_ovr.get("pass_std_multiplier", dec_cfg["pass_std_multiplier"]))`n        fail_mult = float(cat_ovr.get("fail_std_multiplier", dec_cfg["fail_std_multiplier"]))
        test_dir = Path(data_cfg["raw_dir"]) / category / "test"
        gt_root = Path(data_cfg["raw_dir"]) / category / "ground_truth"
        test_imgs, test_masks, y_true, defect_types = [], [], [], []

        for img_path in sorted(test_dir.rglob("*.png")):
            dtype = img_path.parent.name
            is_def = 0 if dtype == "good" else 1
            img = load_and_preprocess_image(img_path, target_size=img_size, normalize=True)
            if is_def:
                mask_path = gt_root / dtype / f"{img_path.stem}_mask.png"
                mask = load_and_preprocess_mask(mask_path, target_size=img_size, binarize=True)[..., 0]
            else:
                mask = np.zeros(img_size, dtype=np.float32)

            test_imgs.append(img)
            test_masks.append(mask)
            y_true.append(is_def)
            defect_types.append(dtype)

        y_true_arr = np.array(y_true, dtype=np.int32)
        test_masks_arr = np.stack(test_masks, axis=0)

        seed_metrics = []
        per_defect_caught: dict[str, list[float]] = defaultdict(list)

        for seed in seeds:
            train_ds, val_ds, _, _ = create_autoencoder_datasets(
                category=category,
                data_dir=data_cfg["raw_dir"],
                batch_size=data_cfg["batch_size"],
                val_split=data_cfg["val_split"],
                random_seed=seed,
            )
            bank = PatchCoreMemoryBank(
                feature_extractor=extractor,
                coreset_ratio=coreset_ratio,
                n_neighbors=pc_cfg["n_neighbors"],
            ).fit(train_ds)

            val_scores = [
                bank.score_image(v_img, pc_cfg["gaussian_kernel"], pc_cfg["gaussian_sigma"])[0]
                for v_batch, _ in val_ds
                for v_img in v_batch.numpy()
            ]
            engine = DecisionEngine(
                pass_std_multiplier=pass_mult,
                fail_std_multiplier=fail_mult,
                category=category,
            ).fit(np.array(val_scores))

            scores, heatmaps, decisions = [], [], []
            for img in test_imgs:
                sc, hmap = bank.score_image(img, pc_cfg["gaussian_kernel"], pc_cfg["gaussian_sigma"])
                res = engine.evaluate(raw_score=sc, heatmap=hmap)
                scores.append(sc)
                heatmaps.append(hmap)
                decisions.append(res.decision)

            scores_arr = np.array(scores, dtype=np.float64)
            heatmaps_arr = np.stack(heatmaps, axis=0)
            decisions_arr = np.array(decisions)

            img_m = compute_image_metrics(y_true_arr, scores_arr, threshold=engine.pass_threshold)
            pix_auroc = compute_pixel_auroc(test_masks_arr, heatmaps_arr)
            pix_iou = compute_pixel_iou(test_masks_arr, heatmaps_arr, threshold=engine.pass_threshold)

            norm_dec = decisions_arr[y_true_arr == 0]
            def_dec = decisions_arr[y_true_arr == 1]

            seed_metrics.append(
                {
                    "seed": seed,
                    "image_auroc": img_m["image_auroc"],
                    "pixel_auroc": pix_auroc,
                    "pixel_iou": pix_iou,
                    "normal_false_fail": int(np.sum(norm_dec == "FAIL")),
                    "defect_caught_rate": float(np.mean(def_dec != "PASS")),
                }
            )

            for dtype in sorted(set(defect_types)):
                if dtype == "good":
                    continue
                idx = [i for i, d in enumerate(defect_types) if d == dtype]
                caught = float(np.mean(decisions_arr[idx] != "PASS"))
                per_defect_caught[dtype].append(caught)

        def _agg(key: str) -> str:
            vals = [m[key] * 100 for m in seed_metrics]
            return f"{np.mean(vals):.2f}% +/- {np.std(vals):.2f}%"

        cat_summary = {
            "image_auroc": _agg("image_auroc"),
            "pixel_auroc": _agg("pixel_auroc"),
            "pixel_iou": _agg("pixel_iou"),
            "defect_caught_rate": _agg("defect_caught_rate"),
            "per_defect_caught": {
                k: f"{np.mean(v)*100:.1f}% +/- {np.std(v)*100:.1f}%"
                for k, v in per_defect_caught.items()
            },
            "runs": seed_metrics,
        }
        report["categories"][category] = cat_summary

        print(
            f"{category:<10} | Img-AUROC: {cat_summary['image_auroc']} | "
            f"Pix-AUROC: {cat_summary['pixel_auroc']} | "
            f"Pix-IoU: {cat_summary['pixel_iou']} | "
            f"Caught: {cat_summary['defect_caught_rate']}"
        )
        for dtype, rate in cat_summary["per_defect_caught"].items():
            print(f"   - {dtype:<20}: {rate}")

    out_path = Path("artifacts/benchmark_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved reproducible benchmark report to {out_path}")
    return report


if __name__ == "__main__":
    run_benchmark()

