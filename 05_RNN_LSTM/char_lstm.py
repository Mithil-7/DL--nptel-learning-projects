"""
Character-Level Language Model with LSTM
=========================================
NPTEL Deep Learning – Project 5

Demonstrates:
  * Character-level tokenisation
  * nn.LSTM with hidden-state management
  * Sequence training with truncated BPTT
  * Temperature-controlled text sampling
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os


# ─── Configuration ────────────────────────────────────────────────────────────

SEQ_LEN    = 100      # context length for each training example
BATCH_SIZE = 64
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT    = 0.3
EPOCHS     = 30
LR         = 3e-3
CLIP_GRAD  = 5.0
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Short illustrative corpus – replace with any .txt file for better results
CORPUS = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes calamity of so long life.
Deep learning is the science of teaching machines to see, hear, and understand.
Neural networks are inspired by the human brain and learn from data.
The NPTEL deep learning course covers neural networks from first principles.
""" * 20   # repeat to create a larger corpus


# ─── Vocabulary ───────────────────────────────────────────────────────────────

def build_vocab(text: str):
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for c, i in c2i.items()}
    return chars, c2i, i2c


# ─── Dataset ─────────────────────────────────────────────────────────────────

def make_sequences(encoded: list, seq_len: int):
    """Return (X, y) pairs for character prediction."""
    X, y = [], []
    for i in range(0, len(encoded) - seq_len - 1, seq_len // 2):
        X.append(encoded[i: i + seq_len])
        y.append(encoded[i + 1: i + seq_len + 1])
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ─── Model ────────────────────────────────────────────────────────────────────

class CharLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout)
        self.fc    = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embed(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size: int, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return h, c


# ─── Training ─────────────────────────────────────────────────────────────────

def train(model, X, y, epochs, lr, batch_size, clip):
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.5)
    n = X.size(0)
    vocab_size = model.fc.out_features
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(n)
        epoch_loss = 0.0
        steps = 0

        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            Xb = X[batch_idx].to(DEVICE)
            yb = y[batch_idx].to(DEVICE)

            hidden = model.init_hidden(Xb.size(0), DEVICE)
            hidden = (hidden[0].detach(), hidden[1].detach())

            optimiser.zero_grad()
            logits, hidden = model(Xb, hidden)
            # logits: (B, T, V)  →  (B*T, V)
            loss = criterion(logits.view(-1, vocab_size), yb.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimiser.step()

            epoch_loss += loss.item()
            steps += 1

        avg_loss = epoch_loss / steps
        history.append(avg_loss)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3d}/{epochs} | loss={avg_loss:.4f} | "
                  f"perplexity={np.exp(avg_loss):.2f}")

    return history


# ─── Sampling ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, seed: str, c2i: dict, i2c: dict,
             length: int = 300, temperature: float = 0.8) -> str:
    model.eval()
    # Prime with seed
    chars = [c2i.get(c, 0) for c in seed]
    x = torch.tensor([chars], dtype=torch.long, device=DEVICE)
    hidden = model.init_hidden(1, DEVICE)
    _, hidden = model(x, hidden)

    result = list(seed)
    last_char = chars[-1]

    for _ in range(length):
        x = torch.tensor([[last_char]], dtype=torch.long, device=DEVICE)
        logits, hidden = model(x, hidden)
        logits = logits[0, 0] / temperature
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        last_char = int(np.random.choice(len(probs), p=probs))
        result.append(i2c[last_char])

    return "".join(result)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # Load external corpus if available
    if os.path.exists("corpus.txt"):
        with open("corpus.txt", encoding="utf-8") as f:
            text = f.read()
        print("Loaded corpus.txt")
    else:
        text = CORPUS
        print(f"Using built-in corpus ({len(text)} characters)")

    chars, c2i, i2c = build_vocab(text)
    vocab_size = len(chars)
    encoded = [c2i[c] for c in text]
    print(f"Vocabulary size: {vocab_size}\n")

    X, y = make_sequences(encoded, SEQ_LEN)
    print(f"Training sequences: {X.shape[0]}\n")

    model = CharLSTM(
        vocab_size=vocab_size,
        embed_dim=64,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    train(model, X, y, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, clip=CLIP_GRAD)

    torch.save({"model_state": model.state_dict(),
                "c2i": c2i, "i2c": i2c}, "char_lstm.pth")
    print("\nCheckpoint saved → char_lstm.pth")

    # Generate sample text
    seed = "To be"
    print(f"\n--- Generated text (seed='{seed}', temp=0.8) ---\n")
    print(generate(model, seed, c2i, i2c, length=400, temperature=0.8))
