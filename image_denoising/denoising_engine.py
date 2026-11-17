import torch


def train_step(denoiser, train_loader, loss_fn, optimizer, device):
    denoiser.train()
    total_loss = 0.0

    for noisy, clean in train_loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        prediction = denoiser(noisy)
        loss = loss_fn(prediction, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def val_step(denoiser, val_loader, loss_fn, device):
    denoiser.eval()
    total_loss = 0.0
    total_images = 0

    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            prediction = denoiser(noisy)
            loss = loss_fn(prediction, clean)

            batch_size = clean.size(0)
            total_loss += loss.item() * batch_size
            total_images += batch_size

    return total_loss / total_images
