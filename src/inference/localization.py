"""
VisionGuard AI - Anomaly Localization and Heatmap Generation

Computes spatial reconstruction error maps, extracts eroded foreground masks
to suppress boundary/background artifacts, generates COLORMAP_JET heatmaps,
and blends semi-transparent inspection overlays.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


def extract_foreground_mask(
    image: np.ndarray,
    erode_pixels: int = 8,
    border_patch: int = 16
) -> np.ndarray:
    """
    Segments the foreground industrial part from the studio background and erodes
    the mask inward to ignore sharp boundary/silhouette reconstruction blur.

    Args:
        image: Array of shape (H, W, 3) in [0.0, 1.0] float32.
        erode_pixels: Number of pixels to erode inward from object edges.
        border_patch: Corner patch size used to estimate background color.

    Returns:
        2D float32 binary mask of shape (H, W) with 1.0 for valid interior
        foreground pixels and 0.0 for background/edge pixels.
    """
    h, w = image.shape[:2]
    p = min(border_patch, h // 4, w // 4)

    # 1. Estimate background RGB color from the 4 image corners
    corners = np.concatenate([
        image[:p, :p].reshape(-1, 3),
        image[:p, -p:].reshape(-1, 3),
        image[-p:, :p].reshape(-1, 3),
        image[-p:, -p:].reshape(-1, 3),
    ], axis=0)
    bg_color = np.median(corners, axis=0)

    # 2. Compute Euclidean color distance from background and threshold via Otsu
    dist = np.linalg.norm(image.astype(np.float32) - bg_color, axis=-1)
    dist_norm = dist / (dist.max() + 1e-8)
    dist_uint8 = (dist_norm * 255.0).astype(np.uint8)

    _, binary = cv2.threshold(dist_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Morphological closing to fill specular highlights inside metallic surfaces
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    # 3. Morphological closing to fill specular highlights inside metallic surfaces
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    # 3b. Fill internal shadow holes so erosion only trims the outer silhouette
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(binary, contours, -1, 255, thickness=cv2.FILLED)


    # 4. Erode inward to strip away outer perimeter and inner-hole edge blur
    if erode_pixels > 0:
        k_size = 2 * erode_pixels + 1
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        binary = cv2.erode(binary, erode_kernel, iterations=1)

    mask = (binary > 0).astype(np.float32)

    # Fallback if image is uniform (e.g., synthetic test arrays) so scoring never empties
    if np.sum(mask) < 0.01 * (h * w):
        return np.ones((h, w), dtype=np.float32)

    return mask


def compute_reconstruction_error_map(
    original: np.ndarray,
    reconstructed: np.ndarray,
    gaussian_kernel: int = 7,
    sigma: float = 1.5,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculates pixel-level absolute difference between original and reconstructed image,
    averages across RGB channels, applies Gaussian smoothing, and optionally masks out
    background/edge regions.

    Args:
        original: Array of shape (H, W, 3) in [0.0, 1.0] float32.
        reconstructed: Array of shape (H, W, 3) in [0.0, 1.0] float32.
        gaussian_kernel: Kernel size for smoothing (must be odd).
        sigma: Gaussian standard deviation.
        mask: Optional 2D array of shape (H, W) in {0.0, 1.0} to zero out edges/background.

    Returns:
        2D array of shape (H, W) with smoothed error intensities in [0.0, 1.0].
    """
    # 1. Absolute difference across all 3 color channels
    diff = np.abs(original.astype(np.float32) - reconstructed.astype(np.float32))

    # 2. Average error across channels: (H, W, 3) -> (H, W)
    error_map = np.mean(diff, axis=-1)

    # 3. Gaussian smoothing to cluster defect regions and suppress grid/sensor noise
    if gaussian_kernel > 1:
        error_map = cv2.GaussianBlur(error_map, (gaussian_kernel, gaussian_kernel), sigma)

    # 4. Apply optional foreground/interior mask
    if mask is not None:
        error_map = error_map * mask.astype(np.float32)

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
    k_percent: float = 0.5,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Calculates a single scalar anomaly score for an entire image from its error map.

    Args:
        error_map: 2D array of shape (H, W).
        method: 'max', 'top_k_mean' (mean of top k% most anomalous pixels), or 'mean'.
        k_percent: Percentage of top pixels to average (default 0.5%).
        mask: Optional 2D binary mask of shape (H, W); if provided, only pixels
            where mask > 0 are included in the calculation.

    Returns:
        Float anomaly score.
    """
    if mask is not None:
        valid_pixels = error_map[mask > 0]
        if valid_pixels.size == 0:
            valid_pixels = error_map.flatten()
    else:
        valid_pixels = error_map.flatten()

    if method == "max":
        return float(np.max(valid_pixels))

    elif method == "top_k_mean":
        # Sort valid pixel errors descending and take the average of the top k%
        flat_errors = np.sort(valid_pixels)[::-1]
        k_pixels = max(1, int(len(flat_errors) * (k_percent / 100.0)))
        return float(np.mean(flat_errors[:k_pixels]))

    elif method == "mean":
        return float(np.mean(valid_pixels))

    else:
        raise ValueError(f"Unknown anomaly scoring method: {method}")