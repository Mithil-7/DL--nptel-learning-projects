# Deep Learning – NPTEL Learning Projects

A collection of deep learning projects built while following the **NPTEL Deep Learning** course.  
Each project focuses on a key concept, starting from scratch with NumPy and progressing to full PyTorch implementations.

---

## Projects

| # | Folder | Topic |
|---|--------|-------|
| 1 | [`01_Linear_Regression`](./01_Linear_Regression) | Linear Regression from scratch (NumPy) |
| 2 | [`02_MLP_from_scratch`](./02_MLP_from_scratch) | Multilayer Perceptron with backpropagation (NumPy) |
| 3 | [`03_MLP_pytorch`](./03_MLP_pytorch) | MLP with PyTorch on MNIST |
| 4 | [`04_CNN`](./04_CNN) | Convolutional Neural Network on CIFAR-10 |
| 5 | [`05_RNN_LSTM`](./05_RNN_LSTM) | LSTM for text / sequence generation |
| 6 | [`06_Autoencoder`](./06_Autoencoder) | Autoencoder for image reconstruction (MNIST) |

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.8+ is recommended.

---

## Project Summaries

### 1 · Linear Regression (NumPy)
Implements gradient descent from scratch to fit a linear model. Demonstrates the core optimisation loop (forward pass → loss → backward pass → weight update) without any framework.

### 2 · MLP from Scratch (NumPy)
A fully-connected neural network built with only NumPy. Implements ReLU activations, softmax output, cross-entropy loss, and manual backpropagation. Trained on a toy XOR / spiral dataset.

### 3 · MLP with PyTorch (MNIST)
Uses `torch.nn` to build a multi-layer perceptron for handwritten-digit classification. Covers DataLoader, training loop, validation accuracy, and saving/loading checkpoints.

### 4 · Convolutional Neural Network (CIFAR-10)
Builds a CNN with convolutional, pooling, batch-norm, and dropout layers. Trained on the CIFAR-10 dataset. Includes a learning-rate scheduler and top-1 accuracy logging.

### 5 · RNN / LSTM (Text Generation)
A character-level language model using PyTorch's `nn.LSTM`. Given a seed string, the model generates new text one character at a time. Demonstrates sequence training, hidden-state management, and temperature sampling.

### 6 · Autoencoder (MNIST)
An encoder–decoder architecture that compresses MNIST digits into a low-dimensional latent space and reconstructs them. Visualises original vs. reconstructed images and the 2-D latent space.

---

## License
MIT
