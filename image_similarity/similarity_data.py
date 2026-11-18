from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_paths = sorted(
            Path(image_dir).glob("*.jpg"),
            key=lambda path: int(path.stem),
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))

        return image_tensor, image_tensor
