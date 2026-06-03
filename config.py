from pathlib import Path

DATASET_PATH = Path("data")
MODEL_PATH   = Path("models")
RESULTS_PATH = Path("results")

CLASSES     = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
NUM_CLASSES = len(CLASSES)

IMG_SIZE  = (224, 224)
BATCH_SIZE = 32
MAX_IMAGES_PER_CLASS = 2000

EPOCHS_FROZEN   = 5
EPOCHS_FINETUNE = 10
FINETUNE_LAYERS = 20

LR_FROZEN   = 1e-4
LR_FINETUNE = 1e-5

RANDOM_SEED      = 42
VALIDATION_SPLIT = 0.15
TEST_SPLIT       = 0.15
