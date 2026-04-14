"""
Convolutional Neural Network – CIFAR-10 Image Classification
============================================================
NPTEL Deep Learning – Project 4

Demonstrates:
  * Conv2d, BatchNorm2d, MaxPool2d, Dropout2d
  * Data augmentation with torchvision.transforms
  * Learning-rate scheduling (CosineAnnealingLR)
  * Training / validation loop with top-1 accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# ─── Configuration ────────────────────────────────────────────────────────────

BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ("plane", "car", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


# ─── Data ─────────────────────────────────────────────────────────────────────

def get_loaders():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_set = datasets.CIFAR10("./data", train=True,  download=True, transform=train_transform)
    test_set  = datasets.CIFAR10("./data", train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=256,        shuffle=False, num_workers=2)
    return train_loader, test_loader


# ─── Model ────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv → BN → ReLU (optionally with max-pool)."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = False):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SimpleCNN(nn.Module):
    """
    Architecture (input 3×32×32):
      ConvBlock(3→64)   → ConvBlock(64→64, pool)   → 16×16
      ConvBlock(64→128) → ConvBlock(128→128, pool)  → 8×8
      ConvBlock(128→256)→ ConvBlock(256→256, pool)  → 4×4
      AdaptiveAvgPool → 256
      FC(256→10)
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   64),
            ConvBlock(64,  64,  pool=True),
            ConvBlock(64,  128),
            ConvBlock(128, 128, pool=True),
            ConvBlock(128, 256),
            ConvBlock(256, 256, pool=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# ─── Training & evaluation ────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimiser, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimiser.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_history(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="train")
    ax1.plot(epochs, val_losses,   label="val")
    ax1.set_title("Loss (CIFAR-10)"); ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(epochs, [a * 100 for a in train_accs], label="train")
    ax2.plot(epochs, [a * 100 for a in val_accs],   label="val")
    ax2.set_title("Accuracy (%)"); ax2.set_xlabel("Epoch"); ax2.legend()

    plt.tight_layout()
    plt.savefig("cnn_cifar10_training.png", dpi=120)
    print("Plot saved → cnn_cifar10_training.png")
    plt.show()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Using device: {DEVICE}\n")

    train_loader, test_loader = get_loaders()

    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimiser = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimiser, criterion)
        va_loss, va_acc = evaluate(model, test_loader, criterion)
        scheduler.step()

        train_losses.append(tr_loss); val_losses.append(va_loss)
        train_accs.append(tr_acc);   val_accs.append(va_acc)

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), "cnn_cifar10_best.pth")

        print(f"Epoch {epoch:>2d}/{EPOCHS} | "
              f"train loss={tr_loss:.4f} acc={tr_acc * 100:.1f}% | "
              f"val   loss={va_loss:.4f} acc={va_acc * 100:.1f}%")

    print(f"\nBest val accuracy: {best_acc * 100:.1f}%")
    print("Best checkpoint saved → cnn_cifar10_best.pth")

    # Per-class accuracy on the test set
    model.load_state_dict(torch.load("cnn_cifar10_best.pth", weights_only=True))
    model.eval()
    class_correct = [0] * 10
    class_total   = [0] * 10
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(1)
            for c in range(10):
                mask = labels == c
                class_correct[c] += (preds[mask] == c).sum().item()
                class_total[c]   += mask.sum().item()
    print("\nPer-class accuracy:")
    for c, name in enumerate(CLASSES):
        acc = 100.0 * class_correct[c] / class_total[c]
        print(f"  {name:<8s}: {acc:.1f}%")

    plot_history(train_losses, val_losses, train_accs, val_accs)
