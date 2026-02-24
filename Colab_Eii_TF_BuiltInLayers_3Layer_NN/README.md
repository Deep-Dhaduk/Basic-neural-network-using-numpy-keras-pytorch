# Colab E-ii: TensorFlow with Built-in Layers - 3-Layer Deep Neural Network

## Framework: TensorFlow (`tf.keras.layers.Dense` + custom `GradientTape` training loop)

---

### Overview

This notebook implements a **3-layer deep neural network** for **non-linear regression** using **TensorFlow's built-in `tf.keras.layers.Dense`** for layer definitions, but still uses a **custom training loop with `tf.GradientTape`** instead of `model.fit()`. This is a middle-ground approach — leveraging Keras layer abstractions while retaining full control over the training process.

### Target Equation

$$y = \sin(x_1) \cdot x_2^2 + \cos(x_3) \cdot x_1 + x_2 \cdot x_3^2$$

### Network Architecture

```
Input(3) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Linear)
```

### Key Features

- **`tf.keras.layers.Dense`** with built-in activation and He initialization (`kernel_initializer='he_normal'`)
- Layers called explicitly in a custom `forward()` function
- **`tf.GradientTape()`** for manual gradient computation — NOT using `model.fit()`
- **`optimizer.apply_gradients()`** for weight updates (Adam optimizer)
- **`tf.keras.losses.MeanSquaredError()`** for loss computation
- **`@tf.function`** decorator for graph-mode compiled execution
- `tf.data.Dataset` for mini-batch data pipeline
- 4D visualization with PCA (scikit-learn)

### What's Different from Colab E-i

| Feature | E-i (From Scratch) | E-ii (Built-in Layers) |
|---------|---------------------|------------------------|
| Weights | Manual `tf.Variable` | `layers.Dense` manages weights |
| Init | Custom He init function | `kernel_initializer='he_normal'` |
| Forward | `tf.einsum` matmul | `dense_layer(x)` calls |
| Training | `GradientTape` | `GradientTape` (same) |
| `model.fit()` | ❌ | ❌ |

### Notebook Sections

1. Imports & setup
2. Synthetic data generation (3 variables) + tf.data.Dataset batching
3. 4D data visualization with PCA dimensionality reduction
4. Model building using `tf.keras.layers.Dense` (individual layers, NOT Sequential)
5. Forward pass function calling Dense layers sequentially
6. Custom training loop with `GradientTape` + Adam optimizer
7. Results visualization (loss curves, predictions vs actual, residuals)
8. Sample predictions table (denormalized to original scale)

---

### 🎥 Video Walkthrough

> A video walkthrough explaining the Colab notebook is uploaded in this same folder. The video explains how Keras Dense layers simplify weight management while the custom GradientTape training loop retains full control over the training process compared to Colab E-i.
