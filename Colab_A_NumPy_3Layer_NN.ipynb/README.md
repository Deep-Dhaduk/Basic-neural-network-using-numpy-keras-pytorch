# Colab A: NumPy-Only 3-Layer Deep Neural Network

## Framework: NumPy (with `tf.einsum` for matrix multiplication)

---

### Overview

This notebook implements a **3-layer deep neural network** for **non-linear regression** using **pure NumPy** — no framework autograd. All forward and backward passes are coded manually from scratch.

### Target Equation

$$y = \sin(x_1) \cdot x_2^2 + \cos(x_3) \cdot x_1 + x_2 \cdot x_3^2$$

### Network Architecture

```
Input(3) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Linear)
```

### Key Features

- Pure NumPy implementation — no framework autograd
- **`tf.einsum('ij,jk->ik', A, B)`** used for all matrix multiplications
- **Manual forward pass** through 3 hidden layers + output
- **Manual backpropagation** implementing full chain rule gradient propagation
- He initialization for weights
- Mini-batch gradient descent
- 4D visualization with PCA (scikit-learn)

### Notebook Sections

1. Imports & setup
2. Synthetic data generation (3 variables)
3. 4D plotting with PCA dimensionality reduction
4. Weight initialization & architecture definition
5. Activation functions and derivatives (ReLU, Linear)
6. Forward pass using `tf.einsum`
7. MSE loss function & derivative
8. Backward pass — full manual chain rule backpropagation
9. Training loop with mini-batch gradient descent
10. Training visualization (loss curves, predictions, residuals)
11. Sample predictions table (denormalized)
12. 4D prediction comparison plots

---

### 🎥 Video Walkthrough

> A video walkthrough explaining the Colab notebook is uploaded in this same folder. The video covers each section of the notebook in detail, including the math behind manual backpropagation and the chain rule derivations.
