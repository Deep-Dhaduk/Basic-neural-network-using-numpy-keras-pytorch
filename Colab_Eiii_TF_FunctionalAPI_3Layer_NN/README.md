# Colab E-iii: TensorFlow Functional API - 3-Layer Deep Neural Network

## Framework: TensorFlow Functional API (`tf.keras.Model` with `Input`/`Output` graph)

---

### Overview

This notebook implements a **3-layer deep neural network** for **non-linear regression** using **TensorFlow's Functional API**. The model is defined as a **Directed Acyclic Graph (DAG)** of layer calls, which supports complex topologies such as multi-input, multi-output, and skip connections. Training uses the standard `model.compile()` + `model.fit()` pipeline.

### Target Equation

$$y = \sin(x_1) \cdot x_2^2 + \cos(x_3) \cdot x_1 + x_2 \cdot x_3^2$$

### Network Architecture

```
Input(3) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Linear)
```

### Key Features

- **`Input(shape=(3,))`** → **`Dense(...)(...)`** → **`Model(inputs, outputs)`** pattern
- DAG-based model definition (supports complex topologies unlike Sequential)
- **`model.compile(optimizer, loss, metrics)`** for configuration
- **`model.fit()`** for training with built-in validation
- Callbacks: **`EarlyStopping`** (patience=100) + **`ReduceLROnPlateau`** (factor=0.5, patience=50)
- `history` object for automatic loss/metric tracking
- Model visualization with `tf.keras.utils.plot_model()`
- He/Kaiming initialization via `kernel_initializer='he_normal'`
- Adam optimizer (lr=0.001) with MSE loss
- MAE tracked as an additional metric
- 4D visualization with PCA (scikit-learn)

### What's Different from E-i and E-ii

| Feature | E-i (Scratch) | E-ii (Dense + Tape) | E-iii (Functional) |
|---------|---------------|---------------------|---------------------|
| Layers | `tf.Variable` | `layers.Dense` | `layers.Dense` |
| Model | No model object | No model object | `Model(inputs, outputs)` |
| Training | `GradientTape` | `GradientTape` | `model.fit()` |
| Callbacks | ❌ | ❌ | ✅ EarlyStopping + LR Scheduler |
| Topology | Fixed | Fixed | DAG (flexible) |

### Notebook Sections

1. Imports & setup
2. Synthetic data generation (3 variables)
3. 4D data visualization with PCA dimensionality reduction
4. Model building using Functional API (`Input` → `Dense` → `Model`)
5. Model compilation (`model.compile`) + training with `model.fit()` and callbacks
6. Results visualization (loss curves, predictions vs actual, residuals)
7. Sample predictions table + MAE curve (denormalized to original scale)

---

### 🎥 Video Walkthrough

> A video walkthrough explaining the Colab notebook is uploaded in this same folder. The video explains how the Functional API defines models as a graph of layer calls, enabling complex topologies, and how `model.compile()` + `model.fit()` simplify the training pipeline compared to the manual GradientTape approach in Colabs E-i and E-ii.
