"""
Multilayer Perceptron from Scratch using NumPy
===============================================
NPTEL Deep Learning – Project 2

Demonstrates:
  * Building a generic dense layer (forward + backward)
  * ReLU activation and Softmax output
  * Cross-entropy loss
  * Mini-batch SGD training
  * Evaluation on a toy spiral dataset
"""

import numpy as np
import matplotlib.pyplot as plt


# ─── Activations ─────────────────────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


# ─── Loss ─────────────────────────────────────────────────────────────────────

def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    """Categorical cross-entropy. y is integer class labels."""
    n = y.shape[0]
    log_p = -np.log(probs[np.arange(n), y] + 1e-12)
    return float(log_p.mean())


def cross_entropy_grad(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of CE loss w.r.t. softmax input (combined)."""
    n = y.shape[0]
    grad = probs.copy()
    grad[np.arange(n), y] -= 1.0
    return grad / n


# ─── Network ─────────────────────────────────────────────────────────────────

class MLP:
    """Two-hidden-layer MLP: input → ReLU → ReLU → softmax."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        # He initialisation: keeps variance stable through ReLU layers
        self.W1 = rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.standard_normal((hidden_dim, hidden_dim)) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.W3 = rng.standard_normal((hidden_dim, output_dim)) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(output_dim)

        # Cache for backward pass
        self._cache: dict = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)
        z3 = a2 @ self.W3 + self.b3
        probs = softmax(z3)
        self._cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2,
                       "z3": z3, "probs": probs}
        return probs

    def backward(self, y: np.ndarray):
        cache = self._cache
        # Output layer gradient
        dz3 = cross_entropy_grad(cache["probs"], y)
        dW3 = cache["a2"].T @ dz3
        db3 = dz3.sum(axis=0)

        # Second hidden layer
        da2 = dz3 @ self.W3.T
        dz2 = da2 * relu_grad(cache["z2"])
        dW2 = cache["a1"].T @ dz2
        db2 = dz2.sum(axis=0)

        # First hidden layer
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_grad(cache["z1"])
        dW1 = cache["X"].T @ dz1
        db1 = dz1.sum(axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
                "W3": dW3, "b3": db3}

    def update(self, grads: dict, lr: float):
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]
        self.W3 -= lr * grads["W3"]
        self.b3 -= lr * grads["b3"]

    def update_adam(self, grads: dict, lr: float,
                    beta1: float = 0.9, beta2: float = 0.999,
                    eps: float = 1e-8):
        """Adam parameter update (bias-corrected)."""
        if not hasattr(self, "_adam_t"):
            self._adam_t = 0
            self._adam_m = {k: np.zeros_like(v) for k, v in grads.items()}
            self._adam_v = {k: np.zeros_like(v) for k, v in grads.items()}

        self._adam_t += 1
        t = self._adam_t
        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t

        for k in grads:
            g = grads[k]
            m = beta1 * self._adam_m[k] + (1.0 - beta1) * g
            v = beta2 * self._adam_v[k] + (1.0 - beta2) * g * g
            self._adam_m[k] = m
            self._adam_v[k] = v
            m_hat = m / bc1
            v_hat = v / bc2
            param = getattr(self, k)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, k, param)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).argmax(axis=1)


# ─── Spiral dataset ───────────────────────────────────────────────────────────

def make_spiral(n_per_class: int = 200, n_classes: int = 3,
                seed: int = 1) -> tuple:
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for k in range(n_classes):
        t = np.linspace(0, 1, n_per_class)
        angle = t * 4 * np.pi + (2 * np.pi * k / n_classes)
        r = t + rng.normal(0, 0.05, n_per_class)
        X_list.append(np.column_stack([r * np.cos(angle), r * np.sin(angle)]))
        y_list.append(np.full(n_per_class, k))
    return np.vstack(X_list), np.hstack(y_list)


# ─── Training ─────────────────────────────────────────────────────────────────

def train(model: MLP, X: np.ndarray, y: np.ndarray,
          lr: float = 3e-3, epochs: int = 500, batch_size: int = 64):
    n = X.shape[0]
    loss_history = []
    rng = np.random.default_rng(42)

    for epoch in range(1, epochs + 1):
        idx = rng.permutation(n)
        epoch_loss = 0.0
        steps = 0

        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            Xb, yb = X[batch_idx], y[batch_idx]

            probs = model.forward(Xb)
            loss = cross_entropy(probs, yb)
            epoch_loss += loss
            steps += 1

            grads = model.backward(yb)
            model.update_adam(grads, lr)

        avg_loss = epoch_loss / steps
        loss_history.append(avg_loss)

        if epoch % 100 == 0 or epoch == 1:
            preds = model.predict(X)
            acc = (preds == y).mean()
            print(f"Epoch {epoch:>4d} | loss={avg_loss:.4f} | train_acc={acc:.3f}")

    return loss_history


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_decision_boundary(model: MLP, X: np.ndarray, y: np.ndarray):
    h = 0.04
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=15, cmap=plt.cm.Spectral, edgecolors="k",
                linewidths=0.3)
    plt.title("MLP Decision Boundary on Spiral Dataset")
    plt.tight_layout()
    plt.savefig("mlp_decision_boundary.png", dpi=120)
    print("Plot saved → mlp_decision_boundary.png")
    plt.show()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X, y = make_spiral(n_per_class=200, n_classes=3)
    print(f"Dataset: {X.shape[0]} samples, {len(np.unique(y))} classes\n")

    model = MLP(input_dim=2, hidden_dim=64, output_dim=3)
    loss_history = train(model, X, y, lr=3e-3, epochs=500, batch_size=64)

    preds = model.predict(X)
    final_acc = (preds == y).mean()
    print(f"\nFinal training accuracy: {final_acc:.3f}")
    plot_decision_boundary(model, X, y)
