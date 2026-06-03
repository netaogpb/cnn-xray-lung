import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0

from config import IMG_SIZE, NUM_CLASSES, FINETUNE_LAYERS


def build_model(weights: str = "imagenet") -> tf.keras.Model:
    """EfficientNetB0 + custom head, base frozen."""
    base = EfficientNetB0(
        weights=weights,
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = False

    model = Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="lung_cnn")

    return model


def unfreeze_top(model: tf.keras.Model, n: int = FINETUNE_LAYERS) -> None:
    """Unfreeze the last n layers of the EfficientNet base for fine-tuning."""
    base = model.layers[0]
    base.trainable = True
    for layer in base.layers[:-n]:
        layer.trainable = False
    print(f"Unfroze last {n} layers of {base.name} for fine-tuning.")
