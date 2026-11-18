import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms

from image_similarity import similarity_config as config
from image_similarity.similarity_model import ConvEncoder


class ImageSearcher:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose(
            [
                transforms.Resize(config.IMG_SIZE),
                transforms.ToTensor(),
            ]
        )

        self.encoder = ConvEncoder().to(self.device)
        self.encoder.load_state_dict(
            torch.load(config.ENCODER_PATH, map_location=self.device, weights_only=True)
        )
        self.encoder.eval()

        embeddings = np.load(config.EMBEDDING_PATH)
        self.image_names = config.IMAGE_NAMES_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if len(self.image_names) != len(embeddings):
            raise ValueError("向量行数和图片文件名数量不一致，请重新建库")

        self.neighbors = NearestNeighbors(metric="cosine").fit(embeddings)

    def search(self, image, top_k=5):
        image_tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            features = self.encoder(image_tensor.to(self.device))
            query = features.flatten(start_dim=1).cpu().numpy()

        distances, indices = self.neighbors.kneighbors(query, n_neighbors=top_k)

        return [
            (config.IMG_PATH / self.image_names[index], float(distance))
            for index, distance in zip(indices[0], distances[0])
        ]


if __name__ == "__main__":
    from argparse import ArgumentParser

    import matplotlib.pyplot as plt
    from PIL import Image

    parser = ArgumentParser()
    parser.add_argument("image_path", help="要查询的图片路径")
    args = parser.parse_args()

    with Image.open(args.image_path) as image:
        query_image = image.convert("RGB")

    searcher = ImageSearcher()
    results = searcher.search(query_image, top_k=5)

    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    axes[0].imshow(query_image)
    axes[0].set_title("Query")
    axes[0].axis("off")

    for ax, (image_path, distance) in zip(axes[1:], results):
        with Image.open(image_path) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"{image_path.name}\nDistance: {distance:.4f}")
        ax.axis("off")

    fig.tight_layout()
    plt.show()
