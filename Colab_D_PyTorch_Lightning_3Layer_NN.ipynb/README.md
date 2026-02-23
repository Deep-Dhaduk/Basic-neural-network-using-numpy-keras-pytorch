# Colab D: PyTorch Lightning 3-Layer Deep Neural Network

## Framework: PyTorch Lightning

---

### Overview

This notebook implements a **3-layer deep neural network** for **non-linear regression** using **PyTorch Lightning**, which abstracts away the boilerplate training loop and provides built-in support for callbacks, logging, and scalability.

### Target Equation

$$y = \sin(x_1) \cdot x_2^2 + \cos(x_3) \cdot x_1 + x_2 \cdot x_3^2$$

### Network Architecture

```
Input(3) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Linear)
```

### Key Features

- **`LightningModule`** subclass with `training_step`, `validation_step`, `configure_optimizers`
- **`LightningDataModule`** for clean data pipeline management
- Built-in `Trainer` with callbacks (e.g., `EarlyStopping`, `ModelCheckpoint`)
- Automatic logging of training/validation metrics
- He/Kaiming initialization
- MSE loss function
- 4D visualization with PCA (scikit-learn)

### Notebook Sections

1. Imports & setup (including Lightning install)
2. Synthetic data generation (3 variables)
3. 4D plotting with PCA dimensionality reduction
4. `LightningDataModule` definition (Dataset + DataLoaders)
5. `LightningModule` model definition
6. Weight initialization (He/Kaiming)
7. Trainer configuration & callbacks
8. Training with `trainer.fit()`
9. Training visualization (loss curves, predictions, residuals)
10. Sample predictions table (denormalized)
11. 4D prediction comparison plots

---

### 🎥 Video Walkthrough

> A video walkthrough explaining the Colab notebook is uploaded in this same folder. The video covers how PyTorch Lightning reduces boilerplate and adds production-ready features like callbacks, logging, and automatic GPU handling.
