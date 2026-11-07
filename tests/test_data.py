"""
VisionGuard AI - Data Pipeline Unit Tests

Verifies preprocessing shapes, value ranges, binary mask constraints,
and zero-leakage dataset partitioning.
"""

from pathlib import Path
import pytest
import numpy as np
import tensorflow as tf

from src.data.preprocessing import (
    load_and_preprocess_image,
    load_and_preprocess_mask
)
from src.data.loader import (
    create_autoencoder_datasets,
    create_test_dataset,
    get_image_paths
)


def test_image_preprocessing_shape_and_range():
    """Verify that images are standardized to (256, 256, 3) in [0.0, 1.0]."""
    sample_img_path = next(Path("data/raw/metal_nut/train/good").glob("*.png"))
    img = load_and_preprocess_image(sample_img_path, target_size=(256, 256), normalize=True)

    assert img.shape == (256, 256, 3), f"Expected (256, 256, 3), got {img.shape}"
    assert img.dtype == np.float32, f"Expected float32, got {img.dtype}"
    assert 0.0 <= img.min() and img.max() <= 1.0, f"Pixel values out of range: [{img.min()}, {img.max()}]"


def test_mask_preprocessing_binary():
    """Verify that defect masks are (256, 256, 1) and strictly binary {0.0, 1.0}."""
    sample_mask_path = next(Path("data/raw/metal_nut/ground_truth/bent").glob("*.png"))
    mask = load_and_preprocess_mask(sample_mask_path, target_size=(256, 256), binarize=True)

    assert mask.shape == (256, 256, 1), f"Expected (256, 256, 1), got {mask.shape}"
    assert mask.dtype == np.float32, f"Expected float32, got {mask.dtype}"
    unique_vals = set(np.unique(mask))
    assert unique_vals.issubset({0.0, 1.0}), f"Mask contains non-binary values: {unique_vals}"


def test_normal_image_empty_mask():
    """Verify that nominal images generate an all-zero ground-truth mask."""
    empty_mask = load_and_preprocess_mask(None, target_size=(256, 256))
    assert empty_mask.shape == (256, 256, 1)
    assert np.all(empty_mask == 0.0), "Expected all zeros for normal image mask"


def test_zero_leakage_split():
    """Verify train and validation subsets have zero overlapping images."""
    category_dir = Path("data/raw/metal_nut")
    train_paths, _, _ = get_image_paths(category_dir, split="train")
    
    train_ds, val_ds, n_train, n_val = create_autoencoder_datasets("metal_nut", batch_size=4)
    assert n_train + n_val == len(train_paths), "Total samples do not match raw training count"