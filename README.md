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
