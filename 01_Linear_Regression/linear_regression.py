"""
Linear Regression from Scratch using NumPy
===========================================
NPTEL Deep Learning – Project 1

Demonstrates:
  * Generating a synthetic dataset
  * Computing MSE loss
  * Gradient descent (batch & stochastic)
  * Plotting the fitted line
"""

import numpy as np
import matplotlib.pyplot as plt


# ─── Data generation ──────────────────────────────────────────────────────────

def make_dataset(n_samples: int = 100, noise: float = 0.3, seed: int = 42):
    """Return (X, y) for y = 2x + 1 + Gaussian noise."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n_samples, 1))
    y = 2.0 * X[:, 0] + 1.0 + rng.normal(0, noise, size=n_samples)
    return X, y


# ─── Loss & gradients ─────────────────────────────────────────────────────────

def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def compute_gradients(X: np.ndarray, y_true: np.ndarray,
                      w: float, b: float):
    """Return (dw, db) for MSE loss."""
    n = len(y_true)
    y_pred = X[:, 0] * w + b
    error = y_pred - y_true
    dw = 2.0 * np.dot(X[:, 0], error) / n
    db = 2.0 * np.mean(error)
    return dw, db


# ─── Training loop ────────────────────────────────────────────────────────────

def train(X: np.ndarray, y: np.ndarray,
          lr: float = 0.1, epochs: int = 200):
    """Gradient descent over the full dataset (batch GD)."""
    w, b = 0.0, 0.0
    history = []

    for epoch in range(1, epochs + 1):
        y_pred = X[:, 0] * w + b
        loss = mse_loss(y_pred, y)
        history.append(loss)

        dw, db = compute_gradients(X, y, w, b)
        w -= lr * dw
        b -= lr * db

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:>4d} | loss={loss:.4f} | w={w:.4f}, b={b:.4f}")

    return w, b, history


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_results(X, y, w, b, history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Scatter + fitted line
    x_line = np.linspace(X.min(), X.max(), 100)
    ax1.scatter(X[:, 0], y, s=15, alpha=0.6, label="data")
    ax1.plot(x_line, w * x_line + b, color="red", linewidth=2,
             label=f"fit: y={w:.2f}x+{b:.2f}")
    ax1.set_title("Linear Regression Fit")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()

    # Loss curve
    ax2.plot(history)
    ax2.set_title("Training Loss (MSE)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE")

    plt.tight_layout()
    plt.savefig("linear_regression_result.png", dpi=120)
    print("Plot saved → linear_regression_result.png")
    plt.show()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X, y = make_dataset(n_samples=150, noise=0.4)
    print(f"Dataset: {X.shape[0]} samples | y = 2x + 1 + noise\n")

    w, b, history = train(X, y, lr=0.1, epochs=300)

    print(f"\nLearned parameters: w={w:.4f} (true=2.0), b={b:.4f} (true=1.0)")
    plot_results(X, y, w, b, history)
