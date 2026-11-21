"""
VisionGuard AI - Zero-Leakage Decision Engine Test-Set Evaluation
Evaluates validation-calibrated PASS / REVIEW / FAIL thresholds on the official MVTec AD test split.
"""

from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.inference.predictor import VisionGuardPredictor


def main() -> None:
    predictor = VisionGuardPredictor()
    header = (
        f"{'Category':<10} | {'AUROC':<7} | {'Normal (PASS/REV/FAIL)':<24} | "
        f"{'Defect (PASS/REV/FAIL)':<24} | {'Defect Caught':<13} | {'False Escape':<12}"
    )
    print(header)
    print("-" * len(header))

    for cat in predictor.categories:
        y_true, scores, decisions = [], [], []
        test_dir = Path(f"data/raw/{cat}/test")
        for img_path in sorted(test_dir.rglob("*.png")):
            is_defect = 0 if img_path.parent.name == "good" else 1
            res = predictor.predict(img_path.read_bytes(), category=cat)
            y_true.append(is_defect)
            scores.append(res["raw_score"])
            decisions.append(res["decision"])

        y_true = np.array(y_true)
        scores = np.array(scores)
        decisions = np.array(decisions)

        auroc = roc_auc_score(y_true, scores) * 100
        norm_dec = decisions[y_true == 0]
        def_dec = decisions[y_true == 1]

        n_pass = int(np.sum(norm_dec == "PASS"))
        n_rev = int(np.sum(norm_dec == "REVIEW"))
        n_fail = int(np.sum(norm_dec == "FAIL"))

        d_pass = int(np.sum(def_dec == "PASS"))
        d_rev = int(np.sum(def_dec == "REVIEW"))
        d_fail = int(np.sum(def_dec == "FAIL"))

        caught_pct = ((d_rev + d_fail) / len(def_dec)) * 100
        escape_pct = (d_pass / len(def_dec)) * 100

        norm_str = f"{n_pass}/{n_rev}/{n_fail} ({len(norm_dec)})"
        def_str = f"{d_pass}/{d_rev}/{d_fail} ({len(def_dec)})"
        print(
            f"{cat:<10} | {auroc:>6.2f}% | {norm_str:<24} | "
            f"{def_str:<24} | {caught_pct:>11.1f}% | {escape_pct:>10.1f}%"
        )


if __name__ == "__main__":
    main()
