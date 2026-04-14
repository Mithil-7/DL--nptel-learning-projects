# Project 5 – Character-Level LSTM (Text Generation)

**Concepts covered:** `nn.LSTM`, embedding layer, truncated BPTT, temperature sampling  
**Framework:** PyTorch  
**Dataset:** Built-in Hamlet excerpt (or provide your own `corpus.txt`)

## Run

```bash
# With built-in corpus
python char_lstm.py

# With a custom corpus (any plain-text file)
cp /path/to/your/book.txt corpus.txt
python char_lstm.py
```

Trains for 30 epochs, saves `char_lstm.pth`, and prints 400 generated characters.

## Architecture

```
Input (B, T) integers
  └─ Embedding(vocab → 64)
  └─ LSTM(64 → 256, 2 layers, dropout=0.3)
  └─ Linear(256 → vocab)
```

## Temperature Sampling

| Temperature | Effect |
|-------------|--------|
| < 0.5 | More deterministic / repetitive |
| 0.8   | Balanced (default) |
| > 1.0 | More random / creative |
