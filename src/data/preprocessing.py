"""
VisionGuard AI - Image and Mask Preprocessing Pipeline

Provides standardized functions for loading, resizing, normalizing,
and binarizing industrial product images and ground-truth defect masks.
"""

from pathlib import Path
from typing import Tuple, Optional, Union
import cv2
import numpy as np


def load_and_preprocess_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = (256, 256),
    normalize: bool = True
) -> np.ndarray:
    """
    Loads an image from disk, standardizes it to 3-channel RGB, resizes it using
    INTER_AREA decimation, and normalizes pixel values to [0.0, 1.0].

    Args:
        image_path: Path to the image file (.png).
        target_size: Desired output dimensions as (width, height).
        normalize: If True, scales pixel intensities from [0, 255] to [0.0, 1.0].

    Returns:
        np.ndarray of shape (height, width, 3) with dtype float32 (if normalized) or uint8.
    """
    path_str = str(image_path)
    # Read image in raw format (preserves native channels)
    img = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise FileNotFoundError(f"Failed to load image from path: {path_str}")

    # Standardize to 3-channel RGB:
    # Case 1: Grayscale images (e.g. screw) with shape (H, W) or (H, W, 1)
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # Case 2: Standard 3-channel BGR from OpenCV
    elif len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Case 3: 4-channel RGBA (alpha transparency)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        raise ValueError(f"Unsupported image shape {img.shape} for file: {path_str}")

    # Resize using INTER_AREA to prevent loss of thin hairline scratches
    if (img.shape[1], img.shape[0]) != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # Normalize to [0.0, 1.0] float32 for neural network inputs
    if normalize:
        img = img.astype(np.float32) / 255.0

    return img


def load_and_preprocess_mask(
    mask_path: Optional[Union[str, Path]] = None,
    target_size: Tuple[int, int] = (256, 256),
    binarize: bool = True
) -> np.ndarray:
    """
    Loads a ground-truth defect mask, resizes it using INTER_NEAREST, adds a channel
    dimension, and ensures strictly binary values {0.0, 1.0}.

    If mask_path is None or does not exist (nominal/good image), returns an all-zero mask.

    Args:
        mask_path: Path to the mask file (*_mask.png), or None for normal images.
        target_size: Desired output dimensions as (width, height).
        binarize: If True, forces all non-zero pixels to 1.0.

    Returns:
        np.ndarray of shape (height, width, 1) with dtype float32.
    """
    # Nominal images have no mask on disk -> generate an all-zero ground truth
    if mask_path is None or not Path(mask_path).exists():
        return np.zeros((target_size[1], target_size[0], 1), dtype=np.float32)

    path_str = str(mask_path)
    mask = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        raise FileNotFoundError(f"Failed to load mask from path: {path_str}")

    # Resize using INTER_NEAREST to preserve strict binary values
    if (mask.shape[1], mask.shape[0]) != target_size:
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    # Convert to float32 binary mask {0.0, 1.0}
    if binarize:
        mask = (mask > 0).astype(np.float32)
    else:
        mask = mask.astype(np.float32) / 255.0

    # Expand dims to (H, W, 1) for standard segmentation tensor contracts
    mask = np.expand_dims(mask, axis=-1)

    return mask


def denormalize_image(img_array: np.ndarray) -> np.ndarray:
    """
    Converts a normalized [0.0, 1.0] float32 image back to [0, 255] uint8 for display.
    """
    clipped = np.clip(img_array, 0.0, 1.0)
    return (clipped * 255.0).astype(np.uint8)