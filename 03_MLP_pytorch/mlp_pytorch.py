"""
Multilayer Perceptron with PyTorch – MNIST Digit Classification
===============================================================
NPTEL Deep Learning – Project 3

Demonstrates:
  * torch.nn.Module subclassing
  * DataLoader with torchvision datasets
  * Training loop with validation
  * Saving and reloading model checkpoints
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# ─── Configuration ────────────────────────────────────────────────────────────

BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Data ─────────────────────────────────────────────────────────────────────

def get_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST("./data", train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=256,        shuffle=False, num_workers=2)
    return train_loader, test_loader


# ─── Model ────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Fully-connected network: 784 → 512 → 256 → 10."""

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


# ─── Plot helpers ─────────────────────────────────────────────────────────────

def plot_history(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="train")
    ax1.plot(epochs, val_losses,   label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(epochs, train_accs, label="train")
    ax2.plot(epochs, val_accs,   label="val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()

    plt.tight_layout()
    plt.savefig("mlp_pytorch_training.png", dpi=120)
    print("Plot saved → mlp_pytorch_training.png")
    plt.show()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Using device: {DEVICE}\n")

    train_loader, test_loader = get_loaders()

    model = MLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=5, gamma=0.5)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimiser, criterion)
        va_loss, va_acc = evaluate(model, test_loader, criterion)
        scheduler.step()

        train_losses.append(tr_loss); val_losses.append(va_loss)
        train_accs.append(tr_acc);   val_accs.append(va_acc)

        print(f"Epoch {epoch:>2d}/{EPOCHS} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val   loss={va_loss:.4f} acc={va_acc:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), "mlp_mnist.pth")
    print("\nCheckpoint saved → mlp_mnist.pth")

    plot_history(train_losses, val_losses, train_accs, val_accs)
