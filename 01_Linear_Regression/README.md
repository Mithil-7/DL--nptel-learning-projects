# Project 1 – Linear Regression from Scratch

**Concepts covered:** gradient descent, MSE loss, weight updates  
**Framework:** NumPy only

## Run

```bash
python linear_regression.py
```

The script will print the training loss every 50 epochs and save a plot
(`linear_regression_result.png`) showing the fitted line and loss curve.

## Key Ideas

| Concept | Formula |
|---------|---------|
| Prediction | `y_pred = w * x + b` |
| MSE Loss | `L = mean((y_pred - y_true)²)` |
| Gradient w.r.t. w | `dL/dw = 2/n · Σ (y_pred - y_true) · x` |
| Gradient w.r.t. b | `dL/db = 2/n · Σ (y_pred - y_true)` |
| Update rule | `w ← w - lr · dL/dw` |
