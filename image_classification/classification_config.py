from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMG_PATH = PROJECT_ROOT / "common" / "dataset"
FASHION_LABELS_PATH = PROJECT_ROOT / "common" / "fashion-labels.csv"
IMG_SIZE = (64, 64)

CLASS_NAMES = {
    0: "上身衣服",
    1: "鞋",
    2: "包",
    3: "下身衣服",
    4: "手表",
}


SEED = 42
TRAIN_RATIO = 0.75
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
MODEL_PATH = PROJECT_ROOT / "image_classification" / "classifier.pt"
