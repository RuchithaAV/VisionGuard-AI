"""
VisionGuard AI - Convolutional Autoencoder Training Engine
"""

import sys
from pathlib import Path

# 1. Add project root to sys.path FIRST
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, Any, Optional
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks

# 2. NOW import from src (it will find 'src' cleanly!)
from src.models.autoencoder import build_autoencoder
from src.data.loader import create_autoencoder_datasets
from src.training.losses import get_loss
from pathlib import Path

# Ensure project root is in sys.path so 'src' is discoverable
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def train_autoencoder(
    category: str = "metal_nut",
    data_dir: str = "data/raw",
    artifacts_dir: str = "artifacts/models",
    target_size: tuple = (256, 256),
    batch_size: int = 16,
    epochs: int = 40,
    learning_rate: float = 1e-3,
    val_split: float = 0.15,
    random_seed: int = 42,
    loss_type: str = "mse",
    ssim_weight: float = 0.5
) -> Dict[str, Any]:
    """
    Trains a Convolutional Autoencoder on normal samples of a specific category.

    Args:
        category: MVTec AD category name (e.g. 'metal_nut', 'screw', 'capsule').
        data_dir: Path to raw dataset root.
        artifacts_dir: Destination directory for saved model weights.
        target_size: Input spatial resolution (width, height).
        batch_size: Mini-batch size.
        epochs: Maximum training epochs.
        learning_rate: Initial Adam optimizer learning rate.
        val_split: Percentage of normal images reserved for validation.
        random_seed: Reproducibility seed.
        loss_type: 'mse' (baseline) or 'ssim_mae' (Milestone 03 structural loss).
        ssim_weight: Weight of the SSIM term when loss_type='ssim_mae'.

    Returns:
        Dict containing trained model, history dictionary, and checkpoint path.
    """
    # Create artifacts directory for saving weights
    save_dir = Path(artifacts_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Baseline keeps its original filename; new loss strategies get a suffix
    # so experiments never overwrite each other.
    suffix = "" if loss_type == "mse" else f"_{loss_type}"
    checkpoint_path = save_dir / f"autoencoder_{category}{suffix}_best.keras"

    print("==================================================")
    print(f" VisionGuard AI - Autoencoder Training: [{category.upper()}]")
    print(f" Target Resolution: {target_size}")
    print(f" Batch Size:        {batch_size}")
    print(f" Initial LR:        {learning_rate}")
    print(f" Loss Strategy:     {loss_type}" + (f" (ssim_weight={ssim_weight})" if loss_type == "ssim_mae" else ""))
    print(f" Model Checkpoint:  {checkpoint_path}")
    print("==================================================\n")

    # 1. Load Data Streams (Zero-leakage split)
    train_ds, val_ds, n_train, n_val = create_autoencoder_datasets(
        category=category,
        data_dir=data_dir,
        target_size=target_size,
        batch_size=batch_size,
        val_split=val_split,
        random_seed=random_seed
    )
    print(f"Dataset partitions: {n_train} training samples | {n_val} validation samples\n")

    # 2. Instantiate Model
    autoencoder, encoder, decoder = build_autoencoder(
        input_shape=(*target_size, 3),
        alpha=0.2
    )

    # 3. Compile with the selected loss (MSE baseline or SSIM+MAE) and Adam Optimizer
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    autoencoder.compile(
        optimizer=optimizer,
        loss=get_loss(loss_type, ssim_weight=ssim_weight),
        metrics=["mae"] # Mean Absolute Error for tracking
    )

    # 4. Engineering Callbacks: Checkpointing, Early Stopping, LR Decay
    training_callbacks = [
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            mode="min",
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            mode="min",
            verbose=1
        )
    ]

    # 5. Execute Training Loop
    history = autoencoder.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=training_callbacks,
        verbose=1
    )

    print(f"\nTraining complete for [{category}]!")
    print(f"Best model weights saved to: {checkpoint_path}")

    return {
        "autoencoder": autoencoder,
        "encoder": encoder,
        "decoder": decoder,
        "history": history.history,
        "checkpoint_path": checkpoint_path
    }


if __name__ == "__main__":
    import sys
    cat = sys.argv[1] if len(sys.argv) > 1 else "metal_nut"
    num_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    train_autoencoder(category=cat, epochs=num_epochs, batch_size=16)