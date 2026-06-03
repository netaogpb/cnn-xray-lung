import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from config import CLASSES, MODEL_PATH, RESULTS_PATH
from data_loader import get_generators

BEST_MODEL = str(MODEL_PATH / "best_model.keras")


def plot_confusion_matrix(cm: np.ndarray, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_training_curves(history: dict, save_path: str) -> None:
    epochs = range(1, len(history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["loss"],     label="Train loss")
    ax1.plot(epochs, history["val_loss"], label="Val loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, history["accuracy"],     label="Train accuracy")
    ax2.plot(epochs, history["val_accuracy"], label="Val accuracy")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.suptitle("Training Curves", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


def evaluate():
    RESULTS_PATH.mkdir(exist_ok=True)

    print(f"Loading model from {BEST_MODEL} ...")
    model = load_model(BEST_MODEL)

    _, _, test_gen = get_generators()

    print("Running predictions on test set ...")
    y_pred_prob = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = test_gen.classes

    # ── Classification report ─────────────────────────────────────────────────
    print("\n=== Classification Report ===")
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    print(report)
    (RESULTS_PATH / "classification_report.txt").write_text(report)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, str(RESULTS_PATH / "confusion_matrix.png"))

    # ── Training curves ───────────────────────────────────────────────────────
    hist_path = RESULTS_PATH / "history.npy"
    if hist_path.exists():
        history = np.load(str(hist_path), allow_pickle=True).item()
        plot_training_curves(history, str(RESULTS_PATH / "training_curves.png"))
    else:
        print("history.npy not found — skipping training curves plot.")

    # ── Overall test accuracy ─────────────────────────────────────────────────
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    evaluate()
