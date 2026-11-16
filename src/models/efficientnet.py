"""
VisionGuard AI - Multi-Scale Deep Feature Extractor (EfficientNetB0)

Extracts mid-level convolutional feature maps from an ImageNet-pretrained
EfficientNetB0 backbone, performs locally-aware neighborhood pooling, and
aligns multi-scale representations into a unified patch embedding tensor.
"""

from typing import List, Tuple
import tensorflow as tf
import keras
from keras import layers


class MultiScaleFeatureExtractor:
    """
    Taps intermediate feature representations from a frozen EfficientNetB0 backbone.

    Args:
        input_shape: Input image dimensions (H, W, C).
        layer_names: Names of intermediate convolutional layers to extract.
            Default: ['block2b_add', 'block3b_add'].
        pool_size: Local neighborhood averaging kernel size.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (256, 256, 3),
        layer_names: List[str] = None,
        pool_size: int = 3,
    ):
        self.input_shape = input_shape
        self.layer_names = layer_names or ["block2b_add", "block3b_add"]
        self.pool_size = pool_size
        self.model = self._build_extractor()

    def _build_extractor(self) -> keras.Model:
        """Constructs a Keras Model with intermediate layer taps and neighborhood pooling."""
        inputs = keras.Input(shape=self.input_shape, name="extractor_input")

        # 1. EfficientNetB0 expects inputs scaled for ImageNet (standard preprocessing)
        # EfficientNet has internal rescaling if using keras.applications.efficientnet.preprocess_input
        # For [0, 1] inputs, multiplying by 255 gives the expected [0, 255] uint8-like float scale
        scaled_inputs = layers.Lambda(lambda x: x * 255.0, name="scale_to_255")(inputs)

        base_model = keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_tensor=scaled_inputs,
        )
        base_model.trainable = False  # Freeze all weights

        # 2. Extract intermediate activation outputs
        outputs = [base_model.get_layer(name).output for name in self.layer_names]

        # 3. Determine target spatial resolution from the deeper layer (e.g. 32x32)
        target_h, target_w = outputs[-1].shape[1:3]

        aligned_features = []
        for feat in outputs:
            # Resize shallower layers to match target grid
            if feat.shape[1:3] != (target_h, target_w):
                feat = layers.Resizing(
                    target_h, target_w, interpolation="bilinear"
                )(feat)

            # Locally-aware neighborhood aggregation (3x3 average pool, stride 1, padding same)
            pooled = layers.AveragePooling2D(
                pool_size=self.pool_size, strides=1, padding="same"
            )(feat)

            aligned_features.append(pooled)

        # 4. Concatenate multi-scale representations along channel dimension
        concatenated = layers.Concatenate(axis=-1, name="patch_embeddings")(aligned_features)

        return keras.Model(inputs=inputs, outputs=concatenated, name="multiscale_feature_extractor")

    def __call__(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Extracts patch embeddings of shape (B, target_H, target_W, total_channels)."""
        return self.model(x, training=False)

    def extract_patch_vectors(self, x: tf.Tensor) -> tf.Tensor:
        """
        Extracts and flattens spatial patch embeddings into a 2D matrix of vectors.

        Returns:
            Tensor of shape (B * target_H * target_W, total_channels).
        """
        embeddings = self(x, training=False)
        # Reshape (B, H, W, C) -> (B * H * W, C)
        b, h, w, c = tf.shape(embeddings)[0], tf.shape(embeddings)[1], tf.shape(embeddings)[2], tf.shape(embeddings)[3]
        return tf.reshape(embeddings, [-1, c])