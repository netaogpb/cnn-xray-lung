import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np

from config import (
    MODEL_PATH, RESULTS_PATH,
    EPOCHS_FROZEN, EPOCHS_FINETUNE,
    LR_FROZEN, LR_FINETUNE,
)
from data_loader import get_generators
from model import build_model, unfreeze_top

MODEL_PATH.mkdir(exist_ok=True)
RESULTS_PATH.mkdir(exist_ok=True)

BEST_MODEL = str(MODEL_PATH / "best_model.keras")


def _callbacks(lr_patience: int = 3, es_patience: int = 5):
    return [
        EarlyStopping(monitor="val_loss", patience=es_patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=lr_patience, min_lr=1e-7, verbose=1),
        ModelCheckpoint(BEST_MODEL, monitor="val_loss", save_best_only=True, verbose=1),
    ]


def train():
    train_gen, val_gen, _ = get_generators()

    # ── Phase 1: frozen base ─────────────────────────────────────────────────
    print("\n=== Phase 1: Training with frozen EfficientNet base ===")
    model = build_model()
    model.compile(
        optimizer=Adam(LR_FROZEN),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history_frozen = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FROZEN,
        callbacks=_callbacks(),
    )

    # ── Phase 2: fine-tuning top layers ──────────────────────────────────────
    print("\n=== Phase 2: Fine-tuning top EfficientNet layers ===")
    unfreeze_top(model)
    model.compile(
        optimizer=Adam(LR_FINETUNE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_finetune = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINETUNE,
        callbacks=_callbacks(lr_patience=2, es_patience=4),
    )

    # ── Save combined history ─────────────────────────────────────────────────
    combined_history = {
        "accuracy":     history_frozen.history["accuracy"]     + history_finetune.history["accuracy"],
        "val_accuracy": history_frozen.history["val_accuracy"] + history_finetune.history["val_accuracy"],
        "loss":         history_frozen.history["loss"]         + history_finetune.history["loss"],
        "val_loss":     history_frozen.history["val_loss"]     + history_finetune.history["val_loss"],
    }
    np.save(str(RESULTS_PATH / "history.npy"), combined_history)
    print(f"\nBest model saved to {BEST_MODEL}")
    print("Training history saved to results/history.npy")


if __name__ == "__main__":
    train()
