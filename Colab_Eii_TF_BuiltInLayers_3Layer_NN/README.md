# Colab E-ii: TensorFlow Built-In Layers 3-Layer Deep Neural Network

## Framework: TensorFlow (`keras.layers.Dense` + custom `GradientTape` loop)

---

### Overview

This notebook implements a **3-layer deep neural network** for **non-linear regression** using **Keras Dense layers** but with a **custom training loop** via `tf.GradientTape`. This is a hybrid approach — leveraging built-in layers while retaining full control over the training process.

### Target Equation

$$y = \sin(x_1) \cdot x_2^2 + \cos(x_3) \cdot x_1 + x_2 \cdot x_3^2$$

### Network Architecture

```
Input(3) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Linear)
```

### Key Features

- **`keras.layers.Dense`** for weight management and forward pass
- **Custom training loop** using `tf.GradientTape` (not `model.fit()`)
- `optimizer.apply_gradients()` for weight updates
- He/Kaiming initialization via Keras initializers
- MSE loss function
- Mini-batch gradient descent
- 4D visualization with PCA (scikit-learn)

### Notebook Sections

1. Imports & setup
2. Synthetic data generation (3 variables)
3. 4D plotting with PCA dimensionality reduction
4. Model construction using `keras.layers.Dense`
5. Optimizer & loss function setup
6. Custom training loop with `tf.GradientTape`
7. Training visualization (loss curves, predictions, residuals)
8. Sample predictions table (denormalized)
9. 4D prediction comparison plots

---

### 🎥 Video Walkthrough

> A video walkthrough explaining the Colab notebook is uploaded in this same folder. The video explains the hybrid approach of combining Keras built-in layers with a custom GradientTape training loop for full training control.
