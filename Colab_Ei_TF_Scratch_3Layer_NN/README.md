# Colab E-i: TensorFlow From Scratch - 3-Layer Deep Neural Network

## Framework: TensorFlow (low-level API only — no `tf.keras` layers)

---

### Overview

This notebook implements a **3-layer deep neural network** for **non-linear regression** using **TensorFlow's lowest-level API** — pure `tf.Variable` weights, `tf.GradientTape` for automatic differentiation, and `tf.einsum` for matrix multiplication. No `tf.keras` layers or `model.fit()` are used.

### Target Equation

$$y = \sin(x_1) \cdot x_2^2 + \cos(x_3) \cdot x_1 + x_2 \cdot x_3^2$$

### Network Architecture

```
Input(3) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Linear)
```

### Key Features

- **NO `tf.keras` layers** — pure `tf.Variable` for all weights and biases
- **`tf.einsum('ij,jk->ik', X, W)`** for all matrix multiplications
- **`tf.nn.relu()`** for activation (low-level, NOT keras activation)
- **`tf.GradientTape()`** for automatic gradient computation
- **`@tf.function`** decorator for compiled/graph-mode execution
- **Manual training loop** — NO `model.fit()`
- He initialization implemented from scratch using `tf.random.normal`
- `tf.data.Dataset` for mini-batch data pipeline
- Adam optimizer via `tf.optimizers.Adam` with `optimizer.apply_gradients()`
- MSE loss computed manually with `tf.reduce_mean(tf.square(...))`
- 4D visualization with PCA (scikit-learn)

### Notebook Sections

1. Imports & setup
2. Synthetic data generation (3 variables) + tf.data.Dataset batching
3. 4D data visualization with PCA dimensionality reduction
4. Weight initialization as `tf.Variable` (He init, NO keras layers)
5. Forward pass with `tf.einsum` + `tf.nn.relu` (`@tf.function` compiled)
6. Training with `tf.GradientTape` — manual gradient loop (NO `model.fit`)
7. Results visualization (loss curves, predictions vs actual, residuals)
8. Sample predictions table (denormalized to original scale)

---

### 🎥 Video Walkthrough

> A video walkthrough explaining the Colab notebook is uploaded in this same folder. The video covers how TensorFlow's low-level API (`tf.Variable`, `tf.GradientTape`, `tf.einsum`) is used to build a neural network from scratch without any Keras abstractions.
