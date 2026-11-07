"""
VisionGuard AI - TensorFlow Data Loader Pipeline

Constructs high-performance tf.data.Dataset pipelines with zero-leakage
train/validation splits and prefetching for training and evaluation.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.data.preprocessing import (
    load_and_preprocess_image,
    load_and_preprocess_mask
)


def get_image_paths(
    category_dir: Path,
    split: str = "train"
) -> Tuple[List[str], List[int], List[Optional[str]]]:
    """
    Collects file paths, binary labels (0=good, 1=defect), and mask paths.

    Args:
        category_dir: Path to specific category (e.g. data/raw/metal_nut).
        split: 'train' or 'test'.

    Returns:
        Tuple of (image_paths, labels, mask_paths).
    """
    img_paths: List[str] = []
    labels: List[int] = []
    mask_paths: List[Optional[str]] = []

    target_dir = category_dir / split
    gt_dir = category_dir / "ground_truth"

    if not target_dir.exists():
        raise FileNotFoundError(f"Directory not found: {target_dir}")

    for subfolder in sorted(target_dir.iterdir()):
        if not subfolder.is_dir():
            continue

        defect_name = subfolder.name
        is_normal = (defect_name == "good")
        files = sorted(list(subfolder.glob("*.png")))

        for img_file in files:
            img_paths.append(str(img_file))
            labels.append(0 if is_normal else 1)

            if is_normal:
                mask_paths.append(None)
            else:
                expected_mask = gt_dir / defect_name / f"{img_file.stem}_mask.png"
                mask_paths.append(str(expected_mask) if expected_mask.exists() else None)

    return img_paths, labels, mask_paths


def create_autoencoder_datasets(
    category: str,
    data_dir: str = "data/raw",
    target_size: Tuple[int, int] = (256, 256),
    batch_size: int = 16,
    val_split: float = 0.15,
    random_seed: int = 42
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """
    Builds train and validation tf.data.Dataset streams for Autoencoder training.
    Strictly partitions normal images to guarantee zero leakage.

    Yields:
        (image_batch, image_batch) for self-reconstruction training.
    """
    category_dir = Path(data_dir) / category
    train_paths, _, _ = get_image_paths(category_dir, split="train")

    if not train_paths:
        raise ValueError(f"No training images found for category '{category}' in {category_dir}")

    # Split normal images into train and validation sets
    train_subset, val_subset = train_test_split(
        train_paths,
        test_size=val_split,
        random_state=random_seed,
        shuffle=True
    )

    def _py_load_image(path_tensor):
        path_str = path_tensor.numpy().decode("utf-8")
        img = load_and_preprocess_image(path_str, target_size=target_size, normalize=True)
        return img

    def _tf_load_image(path_tensor):
        img = tf.py_function(func=_py_load_image, inp=[path_tensor], Tout=tf.float32)
        img.set_shape((*target_size, 3))
        # Autoencoder input equals target: (X, X)
        return img, img

    # 1. Training Pipeline: shuffle, batch, prefetch
    train_ds = (
        tf.data.Dataset.from_tensor_slices(train_subset)
        .shuffle(buffer_size=len(train_subset), seed=random_seed)
        .map(_tf_load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # 2. Validation Pipeline: batch, prefetch (no shuffle needed)
    val_ds = (
        tf.data.Dataset.from_tensor_slices(val_subset)
        .map(_tf_load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, len(train_subset), len(val_subset)


def create_test_dataset(
    category: str,
    data_dir: str = "data/raw",
    target_size: Tuple[int, int] = (256, 256),
    batch_size: int = 16
) -> Tuple[tf.data.Dataset, List[int], int]:
    """
    Builds evaluation tf.data.Dataset for test images (normal + defects)
    and their matching ground-truth masks.

    Yields:
        (image_batch, label_batch, mask_batch)
    """
    category_dir = Path(data_dir) / category
    test_paths, labels, mask_paths = get_image_paths(category_dir, split="test")

    def _py_load_test_sample(img_path_tensor, mask_path_tensor, label_tensor):
        img_str = img_path_tensor.numpy().decode("utf-8")
        mask_raw = mask_path_tensor.numpy().decode("utf-8")
        mask_str = None if mask_raw == "NONE" else mask_raw

        img = load_and_preprocess_image(img_str, target_size=target_size, normalize=True)
        mask = load_and_preprocess_mask(mask_str, target_size=target_size, binarize=True)
        label = int(label_tensor.numpy())
        return img, label, mask

    def _tf_load_test_sample(img_p, mask_p, lbl):
        img, label, mask = tf.py_function(
            func=_py_load_test_sample,
            inp=[img_p, mask_p, lbl],
            Tout=[tf.float32, tf.int32, tf.float32]
        )
        img.set_shape((*target_size, 3))
        label.set_shape(())
        mask.set_shape((*target_size, 1))
        return img, label, mask

    # Replace None masks with 'NONE' string for tensor compatibility
    safe_mask_paths = [p if p is not None else "NONE" for p in mask_paths]

    test_ds = (
        tf.data.Dataset.from_tensor_slices((test_paths, safe_mask_paths, labels))
        .map(_tf_load_test_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return test_ds, labels, len(test_paths)