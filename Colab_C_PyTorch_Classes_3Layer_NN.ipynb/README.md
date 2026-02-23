# Colab C: PyTorch Class-Based 3-Layer Deep Neural Network

## Framework: PyTorch (`nn.Module` + `nn.Linear`)

---

### Overview

This notebook implements a **3-layer deep neural network** for **non-linear regression** using **PyTorch's built-in `nn.Module` and `nn.Linear` layers**. This is the standard, idiomatic way to build neural networks in PyTorch.

### Target Equation

$$y = \sin(x_1) \cdot x_2^2 + \cos(x_3) \cdot x_1 + x_2 \cdot x_3^2$$

### Network Architecture

```
Input(3) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Linear)
```

### Key Features

- Custom `nn.Module` subclass defining the network
- Built-in `nn.Linear` layers with automatic weight management
- `torch.optim` optimizer (e.g., Adam/SGD) for weight updates
- PyTorch `DataLoader` for mini-batch handling
- He/Kaiming initialization
- MSE loss via `nn.MSELoss()`
- 4D visualization with PCA (scikit-learn)

### Notebook Sections

1. Imports & setup
2. Synthetic data generation (3 variables)
3. 4D plotting with PCA dimensionality reduction
4. Custom Dataset & DataLoader setup
5. Model class definition (`nn.Module` subclass)
6. Weight initialization (He/Kaiming)
7. Optimizer & loss function setup
8. Training loop
9. Training visualization (loss curves, predictions, residuals)
10. Sample predictions table (denormalized)
11. 4D prediction comparison plots

---

### 🎥 Video Walkthrough

> A video walkthrough explaining the Colab notebook is uploaded in this same folder. The video explains how PyTorch's class-based approach simplifies neural network construction compared to the raw tensor approach in Colab B.
