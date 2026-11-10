"""
VisionGuard AI - Model Architecture and Localization Unit Tests

Verifies tensor dimensions, forward pass shapes, error map calculations,
and anomaly scoring logic.
"""

import numpy as np
import tensorflow as tf
import pytest

from src.models.autoencoder import build_autoencoder, build_encoder, build_decoder
from src.inference.localization import (
    compute_reconstruction_error_map,
    generate_heatmap_and_overlay,
    calculate_image_anomaly_score
)


def test_autoencoder_architecture_shapes():
    """Verify encoder bottleneck and decoder output dimensions."""
    ae, enc, dec = build_autoencoder(input_shape=(256, 256, 3))
    
    assert enc.output_shape == (None, 16, 16, 256), f"Expected (None, 16, 16, 256), got {enc.output_shape}"
    assert dec.output_shape == (None, 256, 256, 3), f"Expected (None, 256, 256, 3), got {dec.output_shape}"
    assert ae.output_shape == (None, 256, 256, 3), f"Expected (None, 256, 256, 3), got {ae.output_shape}"


def test_autoencoder_forward_pass():
    """Verify that a batch of images passes through and outputs valid probabilities in [0.0, 1.0]."""
    ae, _, _ = build_autoencoder(input_shape=(256, 256, 3))
    dummy_input = tf.random.uniform((2, 256, 256, 3), minval=0.0, maxval=1.0)
    
    output = ae(dummy_input)
    assert output.shape == (2, 256, 256, 3)
    assert tf.reduce_min(output) >= 0.0, "Output has values below 0.0"
    assert tf.reduce_max(output) <= 1.0, "Output has values above 1.0"


def test_reconstruction_error_map():
    """Verify error map generation and Gaussian smoothing dimensions."""
    orig = np.ones((256, 256, 3), dtype=np.float32)
    recon = np.zeros((256, 256, 3), dtype=np.float32)
    
    error_map = compute_reconstruction_error_map(orig, recon, gaussian_kernel=7, sigma=1.5)
    assert error_map.shape == (256, 256), f"Expected (256, 256), got {error_map.shape}"
    assert np.allclose(error_map, 1.0, atol=1e-2), "Error map should be ~1.0 for complete difference"


def test_heatmap_and_overlay_shapes_and_types():
    """Verify heatmap and overlay shapes and uint8 [0, 255] range."""
    orig = np.random.uniform(0.0, 1.0, (256, 256, 3)).astype(np.float32)
    error_map = np.random.uniform(0.0, 1.0, (256, 256)).astype(np.float32)
    
    heatmap, overlay = generate_heatmap_and_overlay(orig, error_map, alpha=0.6)
    assert heatmap.shape == (256, 256, 3)
    assert overlay.shape == (256, 256, 3)
    assert heatmap.dtype == np.uint8
    assert overlay.dtype == np.uint8


def test_anomaly_score_calculation():
    """Verify that top-k anomaly score is strictly positive and scales with error."""
    low_error = np.full((256, 256), 0.05, dtype=np.float32)
    high_error = np.full((256, 256), 0.50, dtype=np.float32)
    
    low_score = calculate_image_anomaly_score(low_error, method="top_k_mean")
    high_score = calculate_image_anomaly_score(high_error, method="top_k_mean")
    
    assert low_score < high_score, "High error map must yield a higher anomaly score"