from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = Path(__file__).resolve().parent

IMG_PATH = PROJECT_ROOT / "common" / "dataset"
IMG_SIZE = (64, 64)

SEED = 42
TRAIN_RATIO = 0.75
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50

ENCODER_PATH = PACKAGE_DIR / "deep_encoder.pt"
DECODER_PATH = PACKAGE_DIR / "deep_decoder.pt"
EMBEDDING_PATH = PACKAGE_DIR / "data_embedding_f.npy"

IMAGE_NAMES_PATH = PACKAGE_DIR / "embedding_image_names.txt"
