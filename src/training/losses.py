"""
VisionGuard AI - Custom Reconstruction Losses

Combines pixel-wise Mean Absolute Error (MAE) with Structural Similarity (SSIM)
so the autoencoder is penalised for blurred edges and lost texture, not just
for average intensity differences. Pure-MSE training produces smooth
reconstructions that blur sharp object borders, which then dominate the
anomaly score (see notebook 02, Section 5).
"""

from typing import Dict, Any
import tensorflow as tf
import keras


@keras.saving.register_keras_serializable(package="visionguard")
class SSIMMAELoss(keras.losses.Loss):
    """
    Weighted sum: loss = ssim_weight * (1 - SSIM) + (1 - ssim_weight) * MAE

    Args:
        ssim_weight: Weight of the structural term in [0, 1]. 0.0 = pure MAE,
            1.0 = pure SSIM.
        max_val: Dynamic range of the images (1.0 for [0, 1] floats).
        filter_size: Gaussian window size for SSIM.
    """

    def __init__(
        self,
        ssim_weight: float = 0.5,
        max_val: float = 1.0,
        filter_size: int = 11,
        name: str = "ssim_mae_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        if not 0.0 <= ssim_weight <= 1.0:
            raise ValueError(f"ssim_weight must be in [0, 1], got {ssim_weight}")
        self.ssim_weight = ssim_weight
        self.max_val = max_val
        self.filter_size = filter_size

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # SSIM is computed per image -> shape (batch,)
        ssim = tf.image.ssim(
            y_true, y_pred, max_val=self.max_val, filter_size=self.filter_size
        )
        ssim_term = 1.0 - ssim

        # MAE averaged over H, W, C -> shape (batch,)
        mae_term = tf.reduce_mean(tf.abs(y_true - y_pred), axis=[1, 2, 3])

        return self.ssim_weight * ssim_term + (1.0 - self.ssim_weight) * mae_term

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "ssim_weight": self.ssim_weight,
            "max_val": self.max_val,
            "filter_size": self.filter_size,
        })
        return config


def get_loss(loss_type: str = "mse", ssim_weight: float = 0.5):
    """
    Returns a Keras loss for the requested training strategy.

    Args:
        loss_type: 'mse' (baseline) or 'ssim_mae' (Milestone 03).
        ssim_weight: Only used for 'ssim_mae'.
    """
    if loss_type == "mse":
        return "mse"
    if loss_type == "ssim_mae":
        return SSIMMAELoss(ssim_weight=ssim_weight)
    raise ValueError(f"Unknown loss_type '{loss_type}'. Use 'mse' or 'ssim_mae'.")