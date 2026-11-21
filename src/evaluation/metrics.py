"""
VisionGuard AI - Standardized Anomaly Detection Evaluation Metrics

Computes image-level and pixel-level detection metrics (Image AUROC, Pixel AUROC,
Average Precision, Precision, Recall, F1-Score, Confusion Matrix) with strict
label validation (0 = normal, 1 = defect; higher anomaly score = defect).
"""

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def validate_binary_labels(labels: np.ndarray, name: str = "labels") -> np.ndarray:
    """Validates that labels are 1D binary {0, 1} with both classes present."""
    arr = np.asarray(labels).astype(np.int32).ravel()
    unique = set(np.unique(arr).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"{name} must contain only 0 (normal) and 1 (defect), got {unique}.")
    if len(unique) < 2:
        raise ValueError(f"{name} must contain both classes (0 and 1) to compute AUROC/AP.")
    return arr


def find_optimal_f1_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> tuple[float, float, float, float]:
    """
    Finds the anomaly score threshold that maximizes F1-score.

    Returns:
        Tuple of (best_threshold, best_f1, precision_at_best, recall_at_best).
    """
    y_true = validate_binary_labels(y_true, "y_true")
    y_scores = np.asarray(y_scores, dtype=np.float64).ravel()

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    # precision_recall_curve returns N+1 precisions/recalls for N thresholds
    f1_scores = (2.0 * precisions[:-1] * recalls[:-1]) / (
        precisions[:-1] + recalls[:-1] + 1e-12
    )

    best_idx = int(np.argmax(f1_scores))
    return (
        float(thresholds[best_idx]),
        float(f1_scores[best_idx]),
        float(precisions[best_idx]),
        float(recalls[best_idx]),
    )


def compute_image_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float | None = None
) -> dict[str, Any]:
    """
    Computes image-level anomaly detection metrics.

    Args:
        y_true: Binary labels of shape (N,) where 0=normal, 1=defect.
        y_scores: Continuous anomaly scores of shape (N,) where higher = more anomalous.
        threshold: Optional decision threshold. If None, computes max-F1 operating point.

    Returns:
        Dictionary with image_auroc, average_precision, threshold, precision,
        recall, f1_score, and confusion_matrix counts (tn, fp, fn, tp).
    """
    y_true = validate_binary_labels(y_true, "y_true")
    y_scores = np.asarray(y_scores, dtype=np.float64).ravel()

    if y_true.shape[0] != y_scores.shape[0]:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_scores {y_scores.shape}")

    image_auroc = float(roc_auc_score(y_true, y_scores))
    avg_precision = float(average_precision_score(y_true, y_scores))

    if threshold is None:
        threshold, f1, precision, recall = find_optimal_f1_threshold(y_true, y_scores)
    else:
        preds = (y_scores >= threshold).astype(np.int32)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        precision = float(tp / (tp + fp + 1e-12))
        recall = float(tp / (tp + fn + 1e-12))
        f1 = float(2.0 * precision * recall / (precision + recall + 1e-12))

    preds = (y_scores >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    return {
        "image_auroc": image_auroc,
        "average_precision": avg_precision,
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compute_pixel_auroc(
    gt_masks: np.ndarray,
    anomaly_maps: np.ndarray
) -> float:
    """
    Computes pixel-level AUROC across all spatial locations in the evaluation set.

    Args:
        gt_masks: Binary ground-truth masks of shape (N, H, W) or (N, H, W, 1) in {0, 1}.
        anomaly_maps: Predicted continuous anomaly heatmaps of shape (N, H, W).

    Returns:
        Float Pixel-level AUROC in [0.0, 1.0].
    """
    masks = np.asarray(gt_masks)
    if masks.ndim == 4 and masks.shape[-1] == 1:
        masks = np.squeeze(masks, axis=-1)

    maps = np.asarray(anomaly_maps, dtype=np.float64)
    if masks.shape != maps.shape:
        raise ValueError(f"Mask shape {masks.shape} must match anomaly_maps shape {maps.shape}")

    flat_masks = (masks > 0.5).astype(np.int32).ravel()
    flat_maps = maps.ravel()

    validate_binary_labels(flat_masks, "gt_masks")
    return float(roc_auc_score(flat_masks, flat_maps))

def compute_pixel_iou(
    y_true_masks: np.ndarray,
    y_pred_heatmaps: np.ndarray,
    threshold: float,
) -> float:
    """Computes pooled pixel-level Intersection-over-Union (IoU) at a fixed threshold."""
    gt = np.asarray(y_true_masks)
    pred = np.asarray(y_pred_heatmaps)
    if gt.ndim == 4 and gt.shape[-1] == 1:
        gt = gt[..., 0]
    if pred.ndim == 4 and pred.shape[-1] == 1:
        pred = pred[..., 0]
    if gt.shape != pred.shape:
        raise ValueError(f"Mask shape {gt.shape} does not match heatmap shape {pred.shape}")

    gt_bin = gt > 0.5
    pred_bin = pred > threshold
    intersection = int(np.logical_and(gt_bin, pred_bin).sum())
    union = int(np.logical_or(gt_bin, pred_bin).sum())
    return float(intersection / union) if union > 0 else 1.0
