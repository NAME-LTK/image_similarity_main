import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from image_similarity import similarity_config as config
from image_similarity.similarity_data import ImageDataset
from image_similarity.similarity_model import ConvEncoder


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize(config.IMG_SIZE),
            transforms.ToTensor(),
        ]
    )
    dataset = ImageDataset(config.IMG_PATH, transform)
    loader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=False, drop_last=False
    )

    encoder = ConvEncoder().to(device)
    encoder.load_state_dict(
        torch.load(config.ENCODER_PATH, map_location=device, weights_only=True)
    )
    encoder.eval()

    batches = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="生成图片向量", unit="批"):
            features = encoder(images.to(device))
            vectors = features.flatten(start_dim=1)
            batches.append(vectors.cpu().numpy())

    embeddings = np.concatenate(batches, axis=0)
    np.save(config.EMBEDDING_PATH, embeddings)

    image_names = [path.name for path in dataset.image_paths]
    config.IMAGE_NAMES_PATH.write_text("\n".join(image_names) + "\n", encoding="utf-8")

    print("向量库形状：", embeddings.shape)
    print("图片文件名数量：", len(image_names))


if __name__ == "__main__":
    main()
