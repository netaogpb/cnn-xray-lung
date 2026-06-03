# CNN Pulmão v2 — EfficientNetB0

Classificador de raio-X de pulmão com 4 classes: **COVID**, **Lung Opacity**, **Normal**, **Viral Pneumonia**.

## Estrutura do projeto

```
├── config.py        # Constantes (paths, hiperparâmetros, classes)
├── data_loader.py   # Carregamento eficiente via flow_from_directory
├── model.py         # Arquitetura EfficientNetB0 + função de fine-tuning
├── train.py         # Treino em 2 fases (frozen → fine-tune)
├── evaluate.py      # Métricas, confusion matrix, curvas de treino
├── predict.py       # Inferência em imagem única
├── requirements.txt
└── data/            # Dataset (não incluído)
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

Baixe o [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
do Kaggle e coloque as pastas dentro de `data/`.

## Como rodar

### 1. Treinar o modelo
```bash
python train.py
```
Treina em 2 fases e salva o melhor modelo em `models/best_model.keras`.

### 2. Avaliar
```bash
python evaluate.py
```
Gera em `results/`:
- `classification_report.txt` — precision, recall, F1 por classe
- `confusion_matrix.png`
- `training_curves.png`

### 3. Prever uma imagem
```bash
python predict.py caminho/para/imagem.png
```

## Arquitetura

- **Base**: EfficientNetB0 (pré-treinado em ImageNet)
- **Head**: GlobalAveragePooling2D → Dense(256) → BatchNorm → Dropout(0.4) → Softmax(4)
- **Fase 1**: base congelada, 5 épocas, lr=1e-4
- **Fase 2**: últimas 20 camadas descongeladas, 10 épocas, lr=1e-5
