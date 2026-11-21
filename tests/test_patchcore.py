"""
VisionGuard AI - Patchcore feature extraction & Memory bank tests
"""

import numpy as np
import pytest
import tensorflow as tf

from src.models.efficientnet import MultiScaleFeatureExtractor
from src.models.feature_memory import PatchCoreMemoryBank


def test_multiscale_feature_extractor_shapes():
    """Verify that images pass through EfficientNetB0 and output (B, 32, 32, 64) patch tensors."""
    dummy_input = tf.random.uniform((2, 256, 256, 3), minval=0.0, maxval=1.0)
    extractor = MultiScaleFeatureExtractor()
    output = extractor(dummy_input)

    assert output.shape == (2, 32, 32, 64)


def test_extract_patch_vectors_flattening():
    """Verify that (B, 32, 32, 64) gets flattened into (B * 1024, 64) 2D vector matrix."""
    dummy_input = tf.random.uniform((2, 256, 256, 3), minval=0.0, maxval=1.0)
    extractor = MultiScaleFeatureExtractor()
    vectors = extractor.extract_patch_vectors(dummy_input)

    # 2 images * 32 * 32 patches = 2048 patch vectors of 64 dimensions each
    assert vectors.shape == (2048, 64)


def test_patchcore_memory_bank_fit_and_score():
    """Verify fitting on a synthetic batch, coreset subsampling, and scoring a test image."""
    # 1. Create a tiny synthetic dataset of 4 images
    dummy_imgs = tf.random.uniform((4, 256, 256, 3), minval=0.0, maxval=1.0)
    dummy_dataset = tf.data.Dataset.from_tensor_slices(dummy_imgs).batch(2)

    # 2. Initialize memory bank with 10% coreset ratio
    memory_bank = PatchCoreMemoryBank(coreset_ratio=0.10)
    memory_bank.fit(dummy_dataset)

    # Total patches = 4 images * 1024 patches = 4096 patches
    # 10% coreset ratio = ~409 patches stored in memory bank
    assert memory_bank.memory_bank is not None
    assert memory_bank.memory_bank.shape[0] == int(4096 * 0.10)
    assert memory_bank.memory_bank.shape[1] == 64

    # 3. Score a single test image
    test_img = np.random.uniform(0.0, 1.0, (256, 256, 3)).astype(np.float32)
    score, heatmap = memory_bank.score_image(test_img)

    # Output score must be a positive float, heatmap must be (256, 256)
    assert isinstance(score, float)
    assert score > 0.0
    assert heatmap.shape == (256, 256)

def test_patchcore_save_and_load_roundtrip(tmp_path):
    """Verify saving and loading a PatchCoreMemoryBank preserves exact scores and metadata."""
    dummy_imgs = tf.random.uniform((3, 256, 256, 3), minval=0.0, maxval=1.0)
    dummy_ds = tf.data.Dataset.from_tensor_slices(dummy_imgs).batch(3)

    bank = PatchCoreMemoryBank(coreset_ratio=0.10).fit(dummy_ds)
    test_img = np.random.uniform(0.0, 1.0, (256, 256, 3)).astype(np.float32)
    orig_score, orig_map = bank.score_image(test_img)

    save_dir = tmp_path / "metal_nut_patchcore"
    bank.save(str(save_dir), metadata={"category": "metal_nut", "val_mean": 18.75})

    loaded_bank, loaded_meta = PatchCoreMemoryBank.load(
        str(save_dir), feature_extractor=bank.extractor
    )
    loaded_score, loaded_map = loaded_bank.score_image(test_img)

    assert loaded_meta["category"] == "metal_nut"
    assert loaded_meta["val_mean"] == pytest.approx(18.75)
    assert loaded_score == pytest.approx(orig_score, rel=1e-6)
    np.testing.assert_allclose(loaded_map, orig_map, rtol=1e-6)