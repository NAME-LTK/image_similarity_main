import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from common.utils import seed_everything
from image_classification import classification_config as config
from image_classification.classification_data import ImageDataset
from image_classification.classification_engine import train_step, val_step
from image_classification.classification_model import Classifier


def main():
    seed_everything(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize(config.IMG_SIZE),
            transforms.ToTensor(),
        ]
    )
    dataset = ImageDataset(config.IMG_PATH, config.FASHION_LABELS_PATH, transform)

    train_size = int(len(dataset) * config.TRAIN_RATIO)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    classifier = Classifier(num_classes=len(config.CLASS_NAMES)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=config.LEARNING_RATE)

    best_val_loss = float("inf")

    for epoch in tqdm(range(1, config.EPOCHS + 1), desc="商品分类训练", unit="轮"):
        train_loss = train_step(classifier, train_loader, loss_fn, optimizer, device)
        val_loss, val_accuracy = val_step(classifier, val_loader, loss_fn, device)

        tqdm.write(
            f"第 {epoch}/{config.EPOCHS} 轮："
            f"训练损失 {train_loss:.4f}，验证损失 {val_loss:.4f}，"
            f"验证准确率 {val_accuracy:.2%}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(classifier.state_dict(), config.MODEL_PATH)
            tqdm.write(f"已保存最佳权重到 {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
