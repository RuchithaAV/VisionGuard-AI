"""Score every screw test image with the deployed screw calibration."""

from collections import Counter, defaultdict
from pathlib import Path

from src.inference.predictor import VisionGuardPredictor


def main() -> None:
    predictor = VisionGuardPredictor()
    engine = predictor.engines["screw"]
    target_size = predictor.category_image_sizes["screw"]
    bank = predictor.banks["screw"]
    counts: Counter[str] = Counter()
    per_type: dict[str, Counter[str]] = defaultdict(Counter)

    print("TYPE,FILE,DECISION,SCORE,Z_SCORE")
    for image_path in sorted(Path("data/raw/screw/test").rglob("*.png")):
        image = predictor._preprocess_bytes(image_path.read_bytes(), target_size)
        raw_score, heatmap = bank.score_image(
            image,
            gaussian_kernel=predictor.pc_cfg["gaussian_kernel"],
            sigma=predictor.pc_cfg["gaussian_sigma"],
        )
        result = engine.evaluate(raw_score, heatmap)
        counts[result.decision] += 1
        defect_type = image_path.parent.name
        per_type[defect_type][result.decision] += 1
        print(
            f"{defect_type},{image_path.name},{result.decision},"
            f"{result.raw_score:.4f},{result.z_score:.4f}"
        )

    print("COUNTS", dict(counts))
    print("PER_TYPE", {key: dict(value) for key, value in per_type.items()})
    print(
        "THRESHOLDS",
        f"pass={engine.pass_threshold:.4f}",
        f"fail={engine.fail_threshold:.4f}",
    )


if __name__ == "__main__":
    main()
