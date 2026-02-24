# Colab E-iv: TensorFlow High-Level API (Sequential) - 3-Layer Deep Neural Network

## Framework: TensorFlow/Keras Sequential API — highest level of abstraction

---

### Overview

This notebook implements a **3-layer deep neural network** for **non-linear regression** using **TensorFlow/Keras `Sequential` API** — the highest and most concise level of abstraction in the TensorFlow ecosystem. It uses the full `model.compile()` → `model.fit()` → `model.evaluate()` → `model.predict()` pipeline with multiple callbacks and **BatchNormalization** for training stability.

### Target Equation

$$y = \sin(x_1) \cdot x_2^2 + \cos(x_3) \cdot x_1 + x_2 \cdot x_3^2$$

### Network Architecture

```
Sequential([
    Input(3),
    Dense(64, ReLU) → BatchNormalization,
    Dense(32, ReLU) → BatchNormalization,
    Dense(16, ReLU) → BatchNormalization,
    Dense(1, Linear)
])
```

### Key Features

- **`keras.Sequential`** — simplest API for linear layer stacks
- **`BatchNormalization`** after each hidden layer for training stability
- Full Keras pipeline: `model.compile()` → `model.fit()` → `model.evaluate()` → `model.predict()`
- Multiple callbacks:
  - **`EarlyStopping`** (patience=100, restore best weights)
  - **`ReduceLROnPlateau`** (factor=0.5, patience=30)
  - **`ModelCheckpoint`** (save best model to `best_model.keras`)
- He/Kaiming initialization via `kernel_initializer='he_normal'`
- Adam optimizer (lr=0.001) with MSE loss
- MAE and MSE tracked as metrics
- `train_test_split` from scikit-learn for data splitting
- **4D prediction comparison** visualization (actual vs predicted vs error)
- 4D visualization with PCA (scikit-learn)
- Most concise implementation across all E-series notebooks

### What's Different from E-i, E-ii, and E-iii

| Feature | E-i (Scratch) | E-ii (Dense + Tape) | E-iii (Functional) | E-iv (Sequential) |
|---------|---------------|---------------------|---------------------|---------------------|
| Layers | `tf.Variable` | `layers.Dense` | `layers.Dense` | `layers.Dense` |
| Model | No model | No model | `Model(in, out)` | `Sequential([...])` |
| Training | `GradientTape` | `GradientTape` | `model.fit()` | `model.fit()` |
| BatchNorm | ❌ | ❌ | ❌ | ✅ |
| Callbacks | ❌ | ❌ | ES + LR | ES + LR + Checkpoint |
| `model.evaluate()` | ❌ | ❌ | ❌ | ✅ |
| Topology | Fixed | Fixed | DAG (flexible) | Linear stack only |

### Notebook Sections

1. Imports & setup
2. Data generation & preparation (`train_test_split` from scikit-learn)
3. 4D data visualization with PCA dimensionality reduction
4. Sequential model construction with `BatchNormalization`
5. Model compilation (`optimizer`, `loss`, `metrics`)
6. Callback configuration (`EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`)
7. Training with `model.fit()` (highest-level API)
8. Model evaluation with `model.evaluate()`
9. Results visualization (loss curves, MAE curves, predictions vs actual, residuals)
10. 4D prediction comparison plots (actual vs predicted vs error in 3D space)
11. Sample predictions table (denormalized to original scale)

---

### 🎥 Video Walkthrough

> A video walkthrough explaining the Colab notebook is uploaded in this same folder. The video covers how the Sequential API provides the most concise implementation, the role of BatchNormalization, and how callbacks automate training management — representing the highest level of abstraction in the TensorFlow series (E-i → E-iv).
