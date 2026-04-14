# Project 3 – MLP with PyTorch on MNIST

**Concepts covered:** `nn.Module`, `DataLoader`, batch-norm, dropout, Adam, LR scheduler  
**Framework:** PyTorch  
**Dataset:** MNIST (auto-downloaded via `torchvision`)

## Run

```bash
python mlp_pytorch.py
```

Downloads MNIST into `./data/`, trains for 10 epochs, saves `mlp_mnist.pth` checkpoint
and `mlp_pytorch_training.png`.

## Architecture

```
Input (1×28×28)
  └─ Flatten → Linear(784→512) → BN → ReLU → Dropout(0.3)
             → Linear(512→256) → BN → ReLU → Dropout(0.3)
             → Linear(256→10)
```

Expected test accuracy: **≥ 98 %** after 10 epochs.
