import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

from config import CLASSES, IMG_SIZE, MODEL_PATH

BEST_MODEL = str(MODEL_PATH / "best_model.keras")


def predict_image(img_path: str) -> dict:
    model = load_model(BEST_MODEL)

    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    arr = preprocess_input(np.expand_dims(np.array(img), axis=0).astype("float32"))

    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    return {
        "predicted_class": CLASSES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {cls: float(p) for cls, p in zip(CLASSES, probs)},
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    result = predict_image(sys.argv[1])

    print(f"\nPredicted class : {result['predicted_class']}")
    print(f"Confidence      : {result['confidence']:.2%}")
    print("\nAll probabilities:")
    for cls, prob in result["probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"  {cls:<20} {prob:6.2%}  {bar}")
