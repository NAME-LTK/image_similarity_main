import torch


def train_step(classifier, train_loader, loss_fn, optimizer, device):
    classifier.train()
    total_loss = 0.0

    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        scores = classifier(images)
        loss = loss_fn(scores, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def val_step(classifier, val_loader, loss_fn, device):
    classifier.eval()
    total_loss = 0.0
    total_correct = 0
    total_images = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            scores = classifier(images)
            loss = loss_fn(scores, targets)
            predictions = scores.argmax(dim=1)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == targets).sum().item()
            total_images += batch_size

    return total_loss / total_images, total_correct / total_images
