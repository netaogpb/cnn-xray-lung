import shutil
import random
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

from config import (
    DATASET_PATH, CLASSES, IMG_SIZE, BATCH_SIZE,
    MAX_IMAGES_PER_CLASS, RANDOM_SEED, VALIDATION_SPLIT, TEST_SPLIT,
)

# ─── helpers ────────────────────────────────────────────────────────────────

def _build_split_dirs(base: Path) -> tuple[Path, Path, Path]:
    """Create train/val/test folder structure under base/splits/."""
    splits = base / "splits"
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            (splits / split / cls).mkdir(parents=True, exist_ok=True)
    return splits / "train", splits / "val", splits / "test"


def prepare_splits(dataset_path: Path = DATASET_PATH) -> tuple[Path, Path, Path]:
    """
    Reads images from dataset_path/<Class>/images/, caps at MAX_IMAGES_PER_CLASS,
    splits into train/val/test and symlinks (or copies) into splits/ sub-dirs.
    Returns (train_dir, val_dir, test_dir).
    """
    splits_root = dataset_path / "splits"
    train_dir = splits_root / "train"
    val_dir   = splits_root / "val"
    test_dir  = splits_root / "test"

    if train_dir.exists():
        print("Splits already exist — skipping.")
        return train_dir, val_dir, test_dir

    _build_split_dirs(dataset_path)

    random.seed(RANDOM_SEED)
    for cls in CLASSES:
        imgs = sorted((dataset_path / cls / "images").glob("*"))
        imgs = imgs[:MAX_IMAGES_PER_CLASS]

        train_val, test = train_test_split(imgs, test_size=TEST_SPLIT, random_state=RANDOM_SEED)
        train, val      = train_test_split(train_val, test_size=VALIDATION_SPLIT / (1 - TEST_SPLIT), random_state=RANDOM_SEED)

        for subset, paths in (("train", train), ("val", val), ("test", test)):
            for p in paths:
                dst = splits_root / subset / cls / p.name
                shutil.copy2(p, dst)

        print(f"{cls}: {len(train)} train / {len(val)} val / {len(test)} test")

    return train_dir, val_dir, test_dir


# ─── generators ─────────────────────────────────────────────────────────────

def _make_gen(datagen: ImageDataGenerator, directory: Path, shuffle: bool):
    return datagen.flow_from_directory(
        str(directory),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        classes=CLASSES,
        shuffle=shuffle,
        seed=RANDOM_SEED,
    )


def get_generators(dataset_path: Path = DATASET_PATH):
    train_dir, val_dir, test_dir = prepare_splits(dataset_path)

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
    )
    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = _make_gen(train_datagen, train_dir, shuffle=True)
    val_gen   = _make_gen(eval_datagen,  val_dir,   shuffle=False)
    test_gen  = _make_gen(eval_datagen,  test_dir,  shuffle=False)

    return train_gen, val_gen, test_gen
