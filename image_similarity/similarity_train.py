import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from common.utils import seed_everything
from image_similarity import similarity_config as config
from image_similarity.similarity_data import ImageDataset
from image_similarity.similarity_engine import train_step, val_step
from image_similarity.similarity_model import ConvDecoder, ConvEncoder


def main():
    seed_everything(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize(config.IMG_SIZE),
            transforms.ToTensor(),
        ]
    )
    dataset = ImageDataset(config.IMG_PATH, transform)

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

    encoder = ConvEncoder().to(device)
    decoder = ConvDecoder().to(device)
    loss_fn = torch.nn.MSELoss()
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.LEARNING_RATE)

    best_val_loss = float("inf")
    for epoch in tqdm(range(1, config.EPOCHS + 1), desc="相似检索训练", unit="轮"):
        train_loss = train_step(
            encoder, decoder, train_loader, loss_fn, optimizer, device
        )
        val_loss = val_step(encoder, decoder, val_loader, loss_fn, device)
        tqdm.write(
            f"第 {epoch}/{config.EPOCHS} 轮："
            f"训练损失 {train_loss:.6f}，验证损失 {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(encoder.state_dict(), config.ENCODER_PATH)
            torch.save(decoder.state_dict(), config.DECODER_PATH)
            tqdm.write("已保存这一轮的编码器和解码器权重")


if __name__ == "__main__":
    main()
