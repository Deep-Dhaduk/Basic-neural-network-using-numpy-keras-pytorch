# Colab E-iv: TensorFlow High-Level API 3-Layer Deep Neural Network

## Framework: TensorFlow (Keras Sequential API — highest level)

---

### Overview

This notebook implements a **3-layer deep neural network** for **non-linear regression** using **TensorFlow's highest-level Sequential API**. This is the simplest and most concise way to build and train a neural network in TensorFlow/Keras using `model.fit()`.

### Target Equation

$$y = \sin(x_1) \cdot x_2^2 + \cos(x_3) \cdot x_1 + x_2 \cdot x_3^2$$

### Network Architecture

```
Input(3) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Linear)
```

### Key Features

- **`tf.keras.Sequential`** model for straightforward layer stacking
- `model.compile()` with optimizer and loss
- `model.fit()` for one-line training with callbacks
- Minimal boilerplate — highest abstraction level
- He/Kaiming initialization via Keras initializers
- MSE loss function
- 4D visualization with PCA (scikit-learn)

### Notebook Sections

1. Imports & setup
2. Synthetic data generation (3 variables)
3. 4D plotting with PCA dimensionality reduction
4. Sequential model definition
5. Model compilation (optimizer, loss, metrics)
6. Training with `model.fit()` and callbacks
7. Training visualization (loss curves, predictions, residuals)
8. Sample predictions table (denormalized)
9. 4D prediction comparison plots

---

### 🎥 Video Walkthrough

> A video walkthrough explaining the Colab notebook is uploaded in this same folder. The video demonstrates the simplest way to build a neural network in TensorFlow using the Sequential API and `model.fit()`, comparing it with the lower-level approaches in Colabs E-i through E-iii.
