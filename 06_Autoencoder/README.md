# Project 6 – Autoencoder on MNIST

**Concepts covered:** encoder-decoder, latent space, reconstruction loss, bottleneck  
**Framework:** PyTorch  
**Dataset:** MNIST (auto-downloaded via `torchvision`)

## Run

```bash
python autoencoder.py
```

Trains for 15 epochs and saves:
- `autoencoder_mnist.pth` – model checkpoint
- `autoencoder_reconstructions.png` – original vs. reconstructed digits
- `autoencoder_latent_space.png` – 2-D latent space coloured by class

## Architecture

```
Encoder:  784 → 512 → ReLU → 128 → ReLU → 2   (latent)
Decoder:    2 → 128 → ReLU → 512 → ReLU → 784 → Sigmoid → 1×28×28
```

## Latent Space Visualisation

With `LATENT_DIM=2` (default) the model forces all information through a 2-D
bottleneck. The scatter plot of the test-set codes shows how well-separated the
10 digit classes are in this compressed space.

Increase `LATENT_DIM` (e.g. 32) for better reconstruction quality at the
cost of a less interpretable latent space.
