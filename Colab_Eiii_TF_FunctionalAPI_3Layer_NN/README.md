# Colab E-iii: TensorFlow Functional API 3-Layer Deep Neural Network

## Framework: TensorFlow (Keras Functional API)

---

### Overview

This notebook implements a **3-layer deep neural network** for **non-linear regression** using **TensorFlow's Functional API**. The model is built by defining an explicit Input/Output computation graph, allowing for more flexible architectures than the Sequential API.

### Target Equation

$$y = \sin(x_1) \cdot x_2^2 + \cos(x_3) \cdot x_1 + x_2 \cdot x_3^2$$

### Network Architecture

```
Input(3) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Linear)
```

### Key Features

- **Keras Functional API** with explicit `Input()` and layer chaining
- `tf.keras.Model(inputs, outputs)` for model construction
- `model.compile()` with optimizer and loss configuration
- `model.fit()` for training with built-in callbacks
- He/Kaiming initialization via Keras initializers
- MSE loss function
- 4D visualization with PCA (scikit-learn)

### Notebook Sections

1. Imports & setup
2. Synthetic data generation (3 variables)
3. 4D plotting with PCA dimensionality reduction
4. Functional API model definition (Input → layers → Output)
5. Model compilation (optimizer, loss, metrics)
6. Training with `model.fit()` and callbacks
7. Training visualization (loss curves, predictions, residuals)
8. Sample predictions table (denormalized)
9. 4D prediction comparison plots

---

### 🎥 Video Walkthrough

> A video walkthrough explaining the Colab notebook is uploaded in this same folder. The video covers the Functional API approach, showing how to define computation graphs and when to prefer this over the Sequential API.
