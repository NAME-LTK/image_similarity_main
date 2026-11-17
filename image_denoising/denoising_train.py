import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from common.utils import seed_everything
from image_denoising.denoising_config import (
    BATCH_SIZE,
    EPOCHS,
    IMG_PATH,
    IMG_SIZE,
    LEARNING_RATE,
    MODEL_PATH,
    NOISE_FACTOR,
    SEED,
    TRAIN_RATIO,
)
from image_denoising.denoising_data import ImageDataset
from image_denoising.denoising_engine import train_step, val_step
from image_denoising.denoising_model import ConvDenoiser


def build_loaders():
    transform = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
        ]
    )
    dataset = ImageDataset(IMG_PATH, transform, NOISE_FACTOR)

    train_size = int(len(dataset) * TRAIN_RATIO)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    return train_loader, val_loader


def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders()

    denoiser = ConvDenoiser().to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    for epoch in tqdm(range(1, EPOCHS + 1), desc="去噪训练", unit="轮"):
        train_loss = train_step(denoiser, train_loader, loss_fn, optimizer, device)
        val_loss = val_step(denoiser, val_loader, loss_fn, device)

        tqdm.write(
            f"第 {epoch}/{EPOCHS} 轮：训练损失 {train_loss:.6f}，验证损失 {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(denoiser.state_dict(), MODEL_PATH)
            tqdm.write(f"已保存最佳权重到 {MODEL_PATH}")


if __name__ == "__main__":
    main()
