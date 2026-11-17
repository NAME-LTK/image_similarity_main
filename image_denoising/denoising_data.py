from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform, noise_factor):
        self.image_paths = sorted(
            Path(image_dir).glob("*.jpg"),
            key=lambda path: int(path.stem),
        )
        self.transform = transform
        self.noise_factor = noise_factor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with Image.open(self.image_paths[idx]) as image:
            clean = self.transform(image.convert("RGB"))

        noise = torch.randn_like(clean) * self.noise_factor
        noisy = (clean + noise).clamp(0.0, 1.0)
        return noisy, clean
