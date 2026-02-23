# Colab B: PyTorch From-Scratch 3-Layer Deep Neural Network

## Framework: PyTorch (raw tensors with `requires_grad=True`)

---

### Overview

This notebook implements a **3-layer deep neural network** for **non-linear regression** using **raw PyTorch tensors** — WITHOUT using built-in `nn.Module` or `nn.Linear` layers. Gradients are computed via PyTorch's autograd engine.

### Target Equation

$$y = \sin(x_1) \cdot x_2^2 + \cos(x_3) \cdot x_1 + x_2 \cdot x_3^2$$

### Network Architecture

```
Input(3) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Linear)
```

### Key Features

- Raw PyTorch tensor operations — no `nn.Module` or `nn.Linear`
- **`torch.einsum('ij,jk->ik', A, B)`** for matrix multiplications
- Manual forward pass with explicit weight matrices and bias vectors
- **PyTorch autograd** for automatic gradient computation (`loss.backward()`)
- Manual weight updates using `torch.no_grad()` context
- He/Kaiming initialization
- Mini-batch gradient descent
- 4D visualization with PCA (scikit-learn)

### Notebook Sections

1. Imports & setup
2. Synthetic data generation (3 variables)
3. 4D plotting with PCA dimensionality reduction
4. Weight & bias initialization with `requires_grad=True`
5. Forward pass function using `torch.einsum`
6. MSE loss computation
7. Training loop with autograd backward pass
8. Training visualization (loss curves, predictions, residuals)
9. Sample predictions table (denormalized)
10. 4D prediction comparison plots

---

### 🎥 Video Walkthrough

> A video walkthrough explaining the Colab notebook is uploaded in this same folder. The video walks through every section, covering how PyTorch tensors and autograd replace the manual backpropagation from the NumPy version.
