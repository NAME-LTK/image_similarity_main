from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, image_dir, labels_path, transform):
        self.image_paths = sorted(
            Path(image_dir).glob("*.jpg"),
            key=lambda path: int(path.stem),
        )
        labels = pd.read_csv(labels_path)
        self.label_by_id = dict(zip(labels["id"], labels["target"]))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_id = int(image_path.stem)
        target = int(self.label_by_id[image_id])

        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))

        return image_tensor, target
