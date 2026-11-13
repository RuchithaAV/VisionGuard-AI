"""
VisionGuard AI - Loss Function Unit Tests
 
Verifies the SSIM + MAE reconstruction loss: zero on identical images, larger
for structural damage, correct weighting, serialization, and training wiring.
"""

import numpy as np
import tensorflow as tf 
import pytest

from src.models.autoencoder import build_autoencoder
from src.training.losses import SSIMMAELoss, get_loss

def _random_batch(n=2, seed=0):
    rng = np.random.default_rng(seed)
    return tf.constant(rng.uniform(0.0, 1.0, (n, 64, 64, 3)).astype(np.float32))

def test_loss_is_zero_for_identical_images():
    x = _random_batch()
    loss = SSIMMAELoss(ssim_weight=0.5)(x,x)
    assert float(loss)  == pytest.approx(0.0, abs=1e-5)

def test_loss_increases_with_structural_damage():
    x = _random_batch()
    noisy = tf.clip_by_value(x + tf.random.normal(x.shape, stddev=0.2, seed=1), 0.0, 1.0)
    slightly_noisy = tf.clip_by_value(x + tf.random.normal(x.shape, stddev=0.02, seed=1), 0.0, 1.0)

    loss_fn = SSIMMAELoss(ssim_weight=0.5)
    assert float(loss_fn(x, noisy)) > float(loss_fn(x, slightly_noisy))

def test_ssim_weight_extremes():
    x = _random_batch()
    y = tf.clip_by_value(x + tf.random.normal(x.shape, stddev = 0.1, seed =2), 0.0, 1.0)
    mae = float(tf.reduce_mean(tf.abs(x - y)))
    ssim = float(tf.reduce_mean(1.0 - tf.image.ssim(x,y, max_val=1.0)))

    assert float(SSIMMAELoss(ssim_weight=0.0)(x,y)) == pytest.approx(mae, rel = 1e-4)
    assert float(SSIMMAELoss(ssim_weight=1.0)(x,y)) == pytest.approx(ssim, rel=1e-4)

def test_invalid_ssim_weight_raises():
    with pytest.raises(ValueError):
        SSIMMAELoss(ssim_weight=1.5)

def test_get_loss_factory():
    assert get_loss("mse") == "mse"
    assert isinstance(get_loss("ssim_mae", ssim_weight=0.7), SSIMMAELoss)
    with pytest.raises(ValueError):
        get_loss("Unknown")

def test_loss_config_roundtrip():
    loss = SSIMMAELoss(ssim_weight=0.3, filter_size=7)
    restored = SSIMMAELoss.from_config(loss.get_config())
    assert restored.ssim_weight == 0.3
    assert restored.filter_size == 7

def test_autoencoder_trains_one_step_with_ssim_loss():
    ae, _, _ = build_autoencoder(input_shape=(256, 256, 3))
    ae.compile(optimizer="adam", loss=get_loss("ssim_mae"))
    x = tf.random.uniform((2, 256, 256, 3), minval=0.0, maxval=1.0)
    history = ae.fit(x, x, epochs=1, batch_size=2, verbose=0)
    assert np.isfinite(history.history["loss"][0])
