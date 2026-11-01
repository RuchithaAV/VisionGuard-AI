"""
Dataset integrity and structure verification for VisionGuard AI (MVTec AD).

Checks:
1. Target categories present (metal_nut, screw, capsule).
2. Proper directory hierarchy: train/good, test/<defect>, ground_truth/<defect>.
3. Checks correspondence between defect images and ground-truth masks.
4. Generates a summary table of images and defect classes.
"""

import os
from pathlib import Path
from typing import Dict, Any
import pandas as pd


TARGET_CATEGORIES = ["metal_nut", "screw", "capsule"]


def verify_category_structure(category_path: Path) -> Dict[str, Any]:
    """Verify that an individual category directory conforms to MVTec AD structure."""
    report = {
        "valid": True,
        "train_good_count": 0,
        "test_good_count": 0,
        "defect_types": {},
        "missing_masks": [],
        "errors": []
    }

    train_good_dir = category_path / "train" / "good"
    if not train_good_dir.exists():
        report["valid"] = False
        report["errors"].append(f"Missing train/good directory at {train_good_dir}")
    else:
        report["train_good_count"] = len(list(train_good_dir.glob("*.png")))

    test_dir = category_path / "test"
    gt_dir = category_path / "ground_truth"

    if not test_dir.exists():
        report["valid"] = False
        report["errors"].append(f"Missing test directory at {test_dir}")
        return report

    for defect_folder in sorted(test_dir.iterdir()):
        if not defect_folder.is_dir():
            continue

        defect_name = defect_folder.name
        img_paths = list(defect_folder.glob("*.png"))
        count = len(img_paths)

        if defect_name == "good":
            report["test_good_count"] = count
        else:
            report["defect_types"][defect_name] = count
            gt_defect_dir = gt_dir / defect_name
            if not gt_defect_dir.exists():
                report["valid"] = False
                report["errors"].append(f"Missing ground truth directory for defect '{defect_name}'")
            else:
                for img_path in img_paths:
                    expected_mask_name = f"{img_path.stem}_mask.png"
                    mask_path = gt_defect_dir / expected_mask_name
                    if not mask_path.exists():
                        report["missing_masks"].append(str(mask_path))

    if report["missing_masks"]:
        report["valid"] = False
        report["errors"].append(f"{len(report['missing_masks'])} defect images lack corresponding ground-truth masks")

    return report


def run_verification(data_dir: str = "data/raw") -> bool:
    """Run verification across all target categories."""
    base_path = Path(data_dir)
    print(f"\n==========================================")
    print(f" VisionGuard AI - Dataset Verification")
    print(f" Target Directory: {base_path.resolve()}")
    print(f"==========================================\n")

    if not base_path.exists():
        print(f"ERROR: Base directory '{base_path}' does not exist.")
        print(f"Please extract the MVTec AD categories into '{base_path}'.")
        return False

    all_valid = True
    summary_rows = []

    for cat in TARGET_CATEGORIES:
        cat_path = base_path / cat
        print(f"Checking [{cat}] ...")
        if not cat_path.exists():
            print(f"  [MISSING] Category directory not found at {cat_path}")
            all_valid = False
            continue

        rep = verify_category_structure(cat_path)
        if rep["valid"]:
            print(f"  [OK] Valid structure.")
            print(f"       Train (Normal): {rep['train_good_count']}")
            print(f"       Test (Normal):  {rep['test_good_count']}")
            print(f"       Defect Types:   {', '.join(f'{k}({v})' for k, v in rep['defect_types'].items())}")
        else:
            print(f"  [FAILED] Issues found:")
            for err in rep["errors"]:
                print(f"       - {err}")
            all_valid = False

        summary_rows.append({
            "Category": cat,
            "Train (Normal)": rep["train_good_count"],
            "Test (Normal)": rep["test_good_count"],
            "Defect Images": sum(rep["defect_types"].values()),
            "Defect Classes": len(rep["defect_types"]),
            "Status": "READY" if rep["valid"] else "INCOMPLETE"
        })

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        print("\n--- Summary Table ---")
        print(df.to_string(index=False))

    print(f"\nOverall Status: {'PASSED - Ready for exploration!' if all_valid else 'ACTION REQUIRED - Complete dataset acquisition.'}\n")
    return all_valid


if __name__ == "__main__":
    import sys
    data_directory = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    run_verification(data_directory)
