# Project 4 – Convolutional Neural Network on CIFAR-10

**Concepts covered:** Conv2d, BatchNorm, MaxPool, data augmentation, AdamW, CosineAnnealingLR  
**Framework:** PyTorch  
**Dataset:** CIFAR-10 (auto-downloaded via `torchvision`)

## Run

```bash
python cnn_cifar10.py
```

Downloads CIFAR-10 into `./data/`, trains for 20 epochs, saves the best checkpoint
(`cnn_cifar10_best.pth`) and a training-curve plot (`cnn_cifar10_training.png`).

## Architecture

```
3×32×32
  └─ ConvBlock(3→64)  → ConvBlock(64→64, pool)    # 16×16
  └─ ConvBlock(64→128)→ ConvBlock(128→128, pool)   # 8×8
  └─ ConvBlock(128→256)→ ConvBlock(256→256, pool)  # 4×4
  └─ AdaptiveAvgPool(1×1) → Flatten(256)
  └─ Dropout(0.3) → Linear(256→10)
```

Expected test accuracy: **≥ 80 %** after 20 epochs with default settings.
