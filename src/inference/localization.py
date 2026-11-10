"""
VisionGuard AI - Anomaly Localization and Heatmap Generation

Computes spatial reconstruction error maps, applies Gaussian noise filtering,
generates COLORMAP_JET heatmaps, and blends semi-transparent inspection overlays.
"""

from typing import Tuple
import cv2
import numpy as np


def compute_reconstruction_error_map(
    original: np.ndarray,
    reconstructed: np.ndarray,
    gaussian_kernel: int = 7,
    sigma: float = 1.5
) -> np.ndarray:
    """
    Calculates pixel-level absolute difference between original and reconstructed image,
    averages across RGB channels, and applies Gaussian smoothing to suppress sensor noise.

    Args:
        original: Array of shape (H, W, 3) in [0.0, 1.0] float32.
        reconstructed: Array of shape (H, W, 3) in [0.0, 1.0] float32.
        gaussian_kernel: Kernel size for smoothing (must be odd).
        sigma: Gaussian standard deviation.

    Returns:
        2D array of shape (H, W) with smoothed error intensities in [0.0, 1.0].
    """
    # 1. Absolute difference across all 3 color channels
    diff = np.abs(original.astype(np.float32) - reconstructed.astype(np.float32))

    # 2. Average error across channels: (H, W, 3) -> (H, W)
    error_map = np.mean(diff, axis=-1)

    # 3. Gaussian smoothing to cluster defect regions and remove isolated noise
    if gaussian_kernel > 1:
        error_map = cv2.GaussianBlur(error_map, (gaussian_kernel, gaussian_kernel), sigma)

    return error_map


def generate_heatmap_and_overlay(
    original: np.ndarray,
    error_map: np.ndarray,
    alpha: float = 0.6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms a continuous 2D error map into an RGB JET heatmap and blends it
    with the original image for visual inspection.

    Args:
        original: Array of shape (H, W, 3) in [0.0, 1.0] float32.
        error_map: 2D array of shape (H, W) in [0.0, 1.0] float32.
        alpha: Blend ratio for original image (1 - alpha for heatmap).

    Returns:
        Tuple of (heatmap_rgb, overlay_rgb) as uint8 arrays [0, 255].
    """
    # 1. Normalize error map to [0, 255] uint8 for color-mapping
    normalized_error = np.clip(error_map / (error_map.max() + 1e-8), 0.0, 1.0)
    error_uint8 = (normalized_error * 255.0).astype(np.uint8)

    # 2. Apply COLORMAP_JET: Blue (cold/normal) -> Yellow -> Red (hot/defect)
    heatmap_bgr = cv2.applyColorMap(error_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # 3. Convert original image to uint8 [0, 255]
    orig_uint8 = (np.clip(original, 0.0, 1.0) * 255.0).astype(np.uint8)

    # 4. Blend original image with heatmap: Overlay = alpha*Original + (1-alpha)*Heatmap
    overlay_rgb = cv2.addWeighted(orig_uint8, alpha, heatmap_rgb, 1.0 - alpha, 0)

    return heatmap_rgb, overlay_rgb


def calculate_image_anomaly_score(
    error_map: np.ndarray,
    method: str = "top_k_mean",
    k_percent: float = 0.5
) -> float:
    """
    Calculates a single scalar anomaly score for an entire image from its error map.

    Args:
        error_map: 2D array of shape (H, W).
        method: 'max' or 'top_k_mean' (mean of top k% most anomalous pixels).
        k_percent: Percentage of top pixels to average (default 0.5%).

    Returns:
        Float anomaly score.
    """
    if method == "max":
        return float(np.max(error_map))

    elif method == "top_k_mean":
        # Sort pixel errors descending and take the average of the top k%
        flat_errors = np.sort(error_map.flatten())[::-1]
        k_pixels = max(1, int(len(flat_errors) * (k_percent / 100.0)))
        return float(np.mean(flat_errors[:k_pixels]))

    elif method == "mean":
        return float(np.mean(error_map))

    else:
        raise ValueError(f"Unknown anomaly scoring method: {method}")