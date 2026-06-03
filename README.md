# Chest X-Ray Lung Disease Classifier — EfficientNetB0

CNN classifier for chest X-ray images across 4 classes: **COVID**, **Lung Opacity**, **Normal**, **Viral Pneumonia**.

Built with transfer learning (EfficientNetB0 pre-trained on ImageNet) and a two-phase training strategy: frozen base first, then fine-tuning the top layers.

## Project structure

```
├── config.py        # Paths, hyperparameters, class names
├── data_loader.py   # Data generators via flow_from_directory
├── model.py         # EfficientNetB0 architecture + fine-tuning helper
├── train.py         # Two-phase training loop
├── evaluate.py      # Metrics, confusion matrix, training curves
├── predict.py       # Single-image inference
├── requirements.txt
└── data/            # Dataset (not included — see below)
    ├── COVID/images/
    ├── Lung_Opacity/images/
    ├── Normal/images/
    └── Viral Pneumonia/images/
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Download the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) from Kaggle and place the class folders inside `data/`.

## Usage

### 1. Train

```bash
python train.py
```

Trains in two phases and saves the best model to `models/best_model.keras`.

### 2. Evaluate

```bash
python evaluate.py
```

Outputs to `results/`:
- `classification_report.txt` — precision, recall, F1 per class
- `confusion_matrix.png`
- `training_curves.png`

### 3. Predict a single image

```bash
python predict.py path/to/image.png
```

## Architecture

| Component | Detail |
|-----------|--------|
| Base model | EfficientNetB0 (ImageNet weights) |
| Head | GlobalAveragePooling2D → Dense(256, ReLU) → BatchNorm → Dropout(0.4) → Softmax(4) |
| Phase 1 | Base frozen — 5 epochs, lr=1e-4 |
| Phase 2 | Top 20 layers unfrozen — 10 epochs, lr=1e-5 |
| Augmentation | Rotation, shift, zoom, horizontal flip, brightness |
| Callbacks | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |

## Training Techniques

### Mini-Batch Gradient Descent

Training uses a **batch size of 32**, meaning weights are updated after every 32 images instead of the full dataset. This balances training stability with memory efficiency and helps the model generalize better.

### Data Augmentation (training only)

Rather than collecting more data, augmentation artificially expands the training set by applying random transforms to each image on the fly:

| Technique | Value |
|-----------|-------|
| Rotation | 20° |
| Horizontal flip | enabled |
| Width / height shift | 10% |
| Zoom | 10% |
| Brightness | 0.8x – 1.2x |

This forces the model to learn features robust to position, orientation, and lighting — important for real-world X-rays that won't always be perfectly aligned.

### Callbacks

| Callback | Config | Purpose |
|----------|--------|---------|
| EarlyStopping | patience=5, restores best weights | Stops training when `val_loss` stops improving, preventing overfitting |
| ReduceLROnPlateau | factor=0.5, patience=3, min_lr=1e-6 | Halves the learning rate when the model plateaus, allowing finer convergence |
| ModelCheckpoint | monitors `val_loss` | Saves the best model to disk automatically |

### Dropout

A Dropout(0.4) layer randomly disables 40% of neurons during training, preventing the model from memorizing training examples and improving generalization.

## Dataset Split

| Split | Size | Purpose |
|-------|------|---------|
| Train | 70% | Model learning |
| Validation | 15% | Hyperparameter tuning and early stopping |
| Test | 15% | Final unbiased evaluation |

Up to **2,000 images per class** are loaded (8,000 total).

## Results

Training improved consistently across epochs with no signs of overfitting:

| Epoch | Train Accuracy | Val Accuracy | Val Loss |
|-------|---------------|--------------|----------|
| 1 | 50.0% | 73.5% | 0.6817 |
| 2 | 73.7% | 78.8% | 0.5784 |
| 3 | 76.9% | 79.5% | 0.5447 |
| 4 | 78.6% | 80.7% | 0.5045 |
| 5 | 80.0% | 80.8% | 0.5028 |

**Test Accuracy: 78.4%**

Real inference example on a COVID X-ray:

```
Predicted class: COVID
Probabilities:
  COVID                0.9102
  Lung_Opacity         0.0700
  Normal               0.0197
  Viral Pneumonia      0.0002
```

> Results obtained training on 2,000 images per class (8,000 total) for 5 epochs. Further improvement expected with more epochs, fine-tuning the base model, or using the full dataset.
