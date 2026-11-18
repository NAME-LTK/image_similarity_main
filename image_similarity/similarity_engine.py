import torch


def train_step(encoder, decoder, train_loader, loss_fn, optimizer, device):
    encoder.train()
    decoder.train()
    total_loss = 0.0

    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        features = encoder(images)
        reconstructed = decoder(features)
        loss = loss_fn(reconstructed, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def val_step(encoder, decoder, val_loader, loss_fn, device):
    encoder.eval()
    decoder.eval()
    total_loss = 0.0
    total_images = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            features = encoder(images)
            reconstructed = decoder(features)
            loss = loss_fn(reconstructed, targets)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_images += batch_size

    return total_loss / total_images
