"""
VisionGuard AI - Evaluation Metrics Unit Tests
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_image_metrics,
    compute_pixel_auroc,
)


def test_perfect_image_separation():
    """Verify AUROC=1.0, AP=1.0, and F1=1.0 when defect scores strictly exceed normal scores."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.25, 0.8, 0.9, 0.95])

    metrics = compute_image_metrics(y_true, y_scores)
    assert metrics["image_auroc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["f1_score"] == pytest.approx(1.0)
    assert metrics["fp"] == 0 and metrics["fn"] == 0


def test_fixed_threshold_confusion_matrix():
    """Verify explicit threshold evaluation returns exact TN, FP, FN, TP counts."""
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([18.0, 22.0, 21.0, 26.0])

    # At threshold 21.5: TN=1 (18.0), FP=1 (22.0), FN=1 (21.0), TP=1 (26.0)
    metrics = compute_image_metrics(y_true, y_scores, threshold=21.5)
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)


def test_pixel_auroc_4d_and_3d_masks():
    """Verify Pixel AUROC handles (N, H, W, 1) masks from loader and aligns spatially."""
    gt_masks = np.zeros((2, 16, 16, 1), dtype=np.float32)
    gt_masks[1, 4:8, 4:8, 0] = 1.0

    pred_maps = np.full((2, 16, 16), 0.1, dtype=np.float32)
    pred_maps[1, 4:8, 4:8] = 0.9

    pixel_auc = compute_pixel_auroc(gt_masks, pred_maps)
    assert pixel_auc == pytest.approx(1.0)


def test_single_class_labels_raise_value_error():
    """Verify invalid single-class labels raise ValueError rather than silently failing."""
    y_true = np.zeros(5, dtype=np.int32)
    y_scores = np.linspace(0.1, 0.5, 5)

    with pytest.raises(ValueError):
        compute_image_metrics(y_true, y_scores)