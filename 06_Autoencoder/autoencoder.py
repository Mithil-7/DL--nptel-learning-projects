"""
Autoencoder for MNIST Image Reconstruction
==========================================
NPTEL Deep Learning – Project 6

Demonstrates:
  * Encoder-decoder architecture with fully-connected layers
  * Latent-space visualisation (2-D)
  * Reconstruction quality (original vs. decoded)
  * MSE reconstruction loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# ─── Configuration ────────────────────────────────────────────────────────────

BATCH_SIZE  = 256
EPOCHS      = 15
LR          = 1e-3
LATENT_DIM  = 2       # set to 2 for easy 2-D latent-space visualisation
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Data ─────────────────────────────────────────────────────────────────────

def get_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = datasets.MNIST("./data", train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, test_loader


# ─── Model ────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 512),        nn.ReLU(),
            nn.Linear(512, 784),        nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


# ─── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimiser, criterion):
    model.train()
    total_loss = 0.0
    for images, _ in loader:
        images = images.to(DEVICE)
        optimiser.zero_grad()
        recon, _ = model(images)
        loss = criterion(recon, images)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    for images, _ in loader:
        images = images.to(DEVICE)
        recon, _ = model(images)
        total_loss += criterion(recon, images).item() * images.size(0)
    return total_loss / len(loader.dataset)


# ─── Visualisation ────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_reconstructions(model, loader, n: int = 8):
    model.eval()
    images, _ = next(iter(loader))
    images = images[:n].to(DEVICE)
    recon, _ = model(images)
    recon = recon.cpu()
    images = images.cpu()

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        axes[0, i].imshow(images[i, 0], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i, 0], cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_title("Original", loc="left")
    axes[1, 0].set_title("Reconstructed", loc="left")
    plt.tight_layout()
    plt.savefig("autoencoder_reconstructions.png", dpi=120)
    print("Plot saved → autoencoder_reconstructions.png")
    plt.show()


@torch.no_grad()
def plot_latent_space(model, loader):
    """2-D scatter of the latent codes coloured by digit class."""
    model.eval()
    all_z, all_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        _, z = model(images)
        all_z.append(z.cpu().numpy())
        all_labels.append(labels.numpy())

    all_z = np.concatenate(all_z)
    all_labels = np.concatenate(all_labels)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(all_z[:, 0], all_z[:, 1],
                          c=all_labels, cmap="tab10", s=2, alpha=0.5)
    plt.colorbar(scatter, ticks=range(10), label="Digit class")
    plt.title("2-D Latent Space (test set)")
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.tight_layout()
    plt.savefig("autoencoder_latent_space.png", dpi=120)
    print("Plot saved → autoencoder_latent_space.png")
    plt.show()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Using device: {DEVICE}\n")

    train_loader, test_loader = get_loaders()

    model     = Autoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimiser, criterion)
        va_loss = evaluate(model, test_loader, criterion)
        print(f"Epoch {epoch:>2d}/{EPOCHS} | train loss={tr_loss:.5f} | val loss={va_loss:.5f}")

    torch.save(model.state_dict(), "autoencoder_mnist.pth")
    print("\nCheckpoint saved → autoencoder_mnist.pth")

    plot_reconstructions(model, test_loader)

    if LATENT_DIM == 2:
        plot_latent_space(model, test_loader)
