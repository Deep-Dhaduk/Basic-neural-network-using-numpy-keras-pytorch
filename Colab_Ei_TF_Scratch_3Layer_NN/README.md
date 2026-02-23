# Colab E-i: TensorFlow From-Scratch 3-Layer Deep Neural Network

## Framework: TensorFlow (low-level — `tf.Variable` + `tf.GradientTape` + `tf.einsum`)

---

### Overview

This notebook implements a **3-layer deep neural network** for **non-linear regression** using **low-level TensorFlow operations** — no Keras layers or high-level APIs. Weights are managed as raw `tf.Variable` objects and gradients are computed via `tf.GradientTape`.

### Target Equation

$$y = \sin(x_1) \cdot x_2^2 + \cos(x_3) \cdot x_1 + x_2 \cdot x_3^2$$

### Network Architecture

```
Input(3) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Linear)
```

### Key Features

- Raw `tf.Variable` weight matrices and bias vectors
- **`tf.einsum('ij,jk->ik', A, B)`** for matrix multiplications
- **`tf.GradientTape`** for automatic gradient computation
- Manual weight updates using `variable.assign_sub()`
- No Keras layers — fully from-scratch TensorFlow
- He initialization for weights
- Mini-batch gradient descent
- MSE loss function
- 4D visualization with PCA (scikit-learn)

### Notebook Sections

1. Imports & setup
2. Synthetic data generation (3 variables)
3. 4D plotting with PCA dimensionality reduction
4. Weight & bias initialization as `tf.Variable`
5. Forward pass function using `tf.einsum`
6. MSE loss computation
7. Training loop with `tf.GradientTape`
8. Training visualization (loss curves, predictions, residuals)
9. Sample predictions table (denormalized)
10. 4D prediction comparison plots

---

### 🎥 Video Walkthrough

> A video walkthrough explaining the Colab notebook is uploaded in this same folder. The video covers how TensorFlow's low-level API (`tf.Variable`, `tf.GradientTape`, `tf.einsum`) is used to build a neural network without any Keras abstractions.
