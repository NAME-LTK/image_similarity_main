from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMG_PATH = PROJECT_ROOT / "common" / "dataset"
IMG_SIZE = (68, 68)  # 预处理后的尺寸，顺序为（高，宽）
NOISE_FACTOR = 0.5

SEED = 42
TRAIN_RATIO = 0.75
BATCH_SIZE = 32


LEARNING_RATE = 0.001
EPOCHS = 50
MODEL_PATH = PROJECT_ROOT / "image_denoising" / "denoiser.pt"
