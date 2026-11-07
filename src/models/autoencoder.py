"""
VisionGuard AI - Convolutional Autoencoder Architecture

Constructs a modular Encoder-Decoder neural network using the Keras Functional API
for unsupervised visual anomaly reconstruction and localization.
"""

from typing import Tuple, Dict, Any
import tensorflow as tf
from tensorflow.keras import layers, Model


def build_encoder(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    alpha: float = 0.2
) -> Model:
    """
    Builds the Encoder sub-model.
    Compresses (256, 256, 3) down to a (16, 16, 256) latent spatial representation.
    """
    inputs = layers.Input(shape=input_shape, name="encoder_input")

    # Block 1: 256x256 -> 128x128
    x = layers.Conv2D(32, kernel_size=3, strides=1, padding="same", name="enc_conv1")(inputs)
    x = layers.BatchNormalization(name="enc_bn1")(x)
    x = layers.LeakyReLU(negative_slope=alpha, name="enc_leaky1")(x)
    x = layers.MaxPooling2D(pool_size=2, padding="same", name="enc_pool1")(x)

    # Block 2: 128x128 -> 64x64
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding="same", name="enc_conv2")(x)
    x = layers.BatchNormalization(name="enc_bn2")(x)
    x = layers.LeakyReLU(negative_slope=alpha, name="enc_leaky2")(x)
    x = layers.MaxPooling2D(pool_size=2, padding="same", name="enc_pool2")(x)

    # Block 3: 64x64 -> 32x32
    x = layers.Conv2D(128, kernel_size=3, strides=1, padding="same", name="enc_conv3")(x)
    x = layers.BatchNormalization(name="enc_bn3")(x)
    x = layers.LeakyReLU(negative_slope=alpha, name="enc_leaky3")(x)
    x = layers.MaxPooling2D(pool_size=2, padding="same", name="enc_pool3")(x)

    # Block 4 (Bottleneck): 32x32 -> 16x16
    x = layers.Conv2D(256, kernel_size=3, strides=1, padding="same", name="enc_conv4")(x)
    x = layers.BatchNormalization(name="enc_bn4")(x)
    x = layers.LeakyReLU(negative_slope=alpha, name="enc_leaky4")(x)
    latent = layers.MaxPooling2D(pool_size=2, padding="same", name="latent_representation")(x)

    return Model(inputs=inputs, outputs=latent, name="encoder")


def build_decoder(
    latent_shape: Tuple[int, int, int] = (16, 16, 256),
    output_channels: int = 3,
    alpha: float = 0.2
) -> Model:
    """
    Builds the Decoder sub-model.
    Upsamples (16, 16, 256) back up to (256, 256, output_channels).
    """
    latent_inputs = layers.Input(shape=latent_shape, name="decoder_input")

    # Block 1: 16x16 -> 32x32
    x = layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same", name="dec_deconv1")(latent_inputs)
    x = layers.BatchNormalization(name="dec_bn1")(x)
    x = layers.LeakyReLU(negative_slope=alpha, name="dec_leaky1")(x)

    # Block 2: 32x32 -> 64x64
    x = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same", name="dec_deconv2")(x)
    x = layers.BatchNormalization(name="dec_bn2")(x)
    x = layers.LeakyReLU(negative_slope=alpha, name="dec_leaky2")(x)

    # Block 3: 64x64 -> 128x128
    x = layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", name="dec_deconv3")(x)
    x = layers.BatchNormalization(name="dec_bn3")(x)
    x = layers.LeakyReLU(negative_slope=alpha, name="dec_leaky3")(x)

    # Block 4 (Final Reconstruction): 128x128 -> 256x256
    # Sigmoid activation guarantees output pixel intensities in [0.0, 1.0]
    outputs = layers.Conv2DTranspose(
        output_channels,
        kernel_size=3,
        strides=2,
        padding="same",
        activation="sigmoid",
        name="reconstruction"
    )(x)

    return Model(inputs=latent_inputs, outputs=outputs, name="decoder")


def build_autoencoder(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    alpha: float = 0.2
) -> Tuple[Model, Model, Model]:
    """
    Connects the Encoder and Decoder into an end-to-end Autoencoder Model.

    Returns:
        Tuple of (autoencoder, encoder, decoder).
    """
    encoder = build_encoder(input_shape=input_shape, alpha=alpha)
    decoder = build_decoder(latent_shape=encoder.output_shape[1:], output_channels=input_shape[-1], alpha=alpha)

    # Connect end-to-end: Input -> Encoder -> Decoder -> Reconstruction
    inputs = layers.Input(shape=input_shape, name="autoencoder_input")
    latent = encoder(inputs)
    reconstruction = decoder(latent)

    autoencoder = Model(inputs=inputs, outputs=reconstruction, name="conv_autoencoder")

    return autoencoder, encoder, decoder