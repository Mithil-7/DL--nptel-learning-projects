# Project 2 – Multilayer Perceptron from Scratch (NumPy)

**Concepts covered:** backpropagation, ReLU, softmax, cross-entropy, mini-batch Adam optimizer  
**Framework:** NumPy only  
**Dataset:** Synthetic 3-class spiral

## Run

```bash
python mlp_numpy.py
```

Trains for 500 epochs and saves a decision-boundary plot (`mlp_decision_boundary.png`).

## Architecture

```
Input (2) → Dense(64) → ReLU → Dense(64) → ReLU → Dense(3) → Softmax
```

## Key Implementation Details

| Component | Notes |
|-----------|-------|
| Forward pass | Layer-by-layer matrix multiply + activation |
| Backward pass | Chain rule applied manually per layer |
| Loss | Categorical cross-entropy |
| Optimiser | Adam (from scratch: momentum + adaptive LR) |
