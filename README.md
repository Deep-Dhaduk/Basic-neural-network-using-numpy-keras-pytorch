# 3-Layer Deep Neural Network for Non-Linear Regression

## Using NumPy, PyTorch, PyTorch Lightning, and TensorFlow

---

## 📋 Project Overview

This repository implements a **3-layer deep neural network** for **non-linear regression with 3 input variables** using multiple frameworks. Each Colab notebook demonstrates a different approach — from pure NumPy with manual backpropagation to high-level TensorFlow/Keras APIs.

### Target Non-Linear Equation

$$y = \sin(x_1) \cdot x_2^2 + \cos(x_3) \cdot x_1 + x_2 \cdot x_3^2$$

### Network Architecture (All Notebooks)

```
Input(3) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Linear)
```

- **3 hidden layers** (64 → 32 → 16 neurons)
- **ReLU** activation for hidden layers
- **He/Kaiming** weight initialization
- **MSE** loss function
- **4D visualization** using PCA dimensionality reduction (scikit-learn)

---

## 📂 File Structure

Each notebook lives in its own folder along with a **README.md** and a **video walkthrough** explaining the Colab.

```
├── README.md                                          ← This file
├── Colab_A_NumPy_3Layer_NN.ipynb/
│   ├── Colab_A_NumPy_3Layer_NN.ipynb
│   ├── README.md
│   └── 🎥 Video walkthrough
├── Colab_B_PyTorch_Scratch_3Layer_NN.ipynb/
│   ├── Colab_B_PyTorch_Scratch_3Layer_NN.ipynb
│   ├── README.md
│   └── 🎥 Video walkthrough
├── Colab_C_PyTorch_Classes_3Layer_NN.ipynb/
│   ├── Colab_C_PyTorch_Classes_3Layer_NN.ipynb
│   ├── README.md
│   └── 🎥 Video walkthrough
├── Colab_D_PyTorch_Lightning_3Layer_NN.ipynb/
│   ├── Colab_D_PyTorch_Lightning_3Layer_NN.ipynb
│   ├── README.md
│   └── 🎥 Video walkthrough
├── Colab_Ei_TF_Scratch_3Layer_NN/
│   ├── Colab_Ei_TF_Scratch_3Layer_NN.ipynb
│   ├── README.md
│   └── 🎥 Video walkthrough
├── Colab_Eii_TF_BuiltInLayers_3Layer_NN/
│   ├── Colab_Eii_TF_BuiltInLayers_3Layer_NN.ipynb
│   ├── README.md
│   └── 🎥 Video walkthrough
├── Colab_Eiii_TF_FunctionalAPI_3Layer_NN/
│   ├── Colab_Eiii_TF_FunctionalAPI_3Layer_NN.ipynb
│   ├── README.md
│   └── 🎥 Video walkthrough
└── Colab_Eiv_TF_HighLevelAPI_3Layer_NN/
    ├── Colab_Eiv_TF_HighLevelAPI_3Layer_NN.ipynb
    ├── README.md
    └── 🎥 Video walkthrough
```

| Folder | Framework | Description |
|--------|-----------|-------------|
| `Colab_A_NumPy_3Layer_NN.ipynb/` | NumPy + tf.einsum | From-scratch NN with manual backprop & chain rule |
| `Colab_B_PyTorch_Scratch_3Layer_NN.ipynb/` | PyTorch (raw tensors) | From-scratch NN WITHOUT built-in layers |
| `Colab_C_PyTorch_Classes_3Layer_NN.ipynb/` | PyTorch (nn.Module) | Class-based NN using built-in PyTorch modules |
| `Colab_D_PyTorch_Lightning_3Layer_NN.ipynb/` | PyTorch Lightning | Lightning framework with DataModule & callbacks |
| `Colab_Ei_TF_Scratch_3Layer_NN/` | TensorFlow (low-level) | tf.Variable + tf.GradientTape + tf.einsum |
| `Colab_Eii_TF_BuiltInLayers_3Layer_NN/` | TensorFlow (Dense layers) | keras.layers.Dense + custom GradientTape loop |
| `Colab_Eiii_TF_FunctionalAPI_3Layer_NN/` | TensorFlow (Functional) | Functional API with Input/Output graph |
| `Colab_Eiv_TF_HighLevelAPI_3Layer_NN/` | TensorFlow (Sequential) | Highest-level API with model.fit() |

---

## 🎥 Video Walkthroughs

> **Each notebook has a corresponding video code walkthrough uploaded in the same folder, explaining every section of the Colab.** Each folder also contains its own README with detailed descriptions.

| Colab | Video Location | Folder README |
|-------|---------------|---------------|
| **Colab A** - NumPy from Scratch | 📹 Video in `Colab_A_NumPy_3Layer_NN.ipynb/` | [README](Colab_A_NumPy_3Layer_NN.ipynb/README.md) |
| **Colab B** - PyTorch from Scratch | 📹 Video in `Colab_B_PyTorch_Scratch_3Layer_NN.ipynb/` | [README](Colab_B_PyTorch_Scratch_3Layer_NN.ipynb/README.md) |
| **Colab C** - PyTorch Classes | 📹 Video in `Colab_C_PyTorch_Classes_3Layer_NN.ipynb/` | [README](Colab_C_PyTorch_Classes_3Layer_NN.ipynb/README.md) |
| **Colab D** - PyTorch Lightning | 📹 Video in `Colab_D_PyTorch_Lightning_3Layer_NN.ipynb/` | [README](Colab_D_PyTorch_Lightning_3Layer_NN.ipynb/README.md) |
| **Colab E-i** - TF from Scratch | 📹 Video in `Colab_Ei_TF_Scratch_3Layer_NN/` | [README](Colab_Ei_TF_Scratch_3Layer_NN/README.md) |
| **Colab E-ii** - TF Built-in Layers | 📹 Video in `Colab_Eii_TF_BuiltInLayers_3Layer_NN/` | [README](Colab_Eii_TF_BuiltInLayers_3Layer_NN/README.md) |
| **Colab E-iii** - TF Functional API | 📹 Video in `Colab_Eiii_TF_FunctionalAPI_3Layer_NN/` | [README](Colab_Eiii_TF_FunctionalAPI_3Layer_NN/README.md) |
| **Colab E-iv** - TF High-Level API | 📹 Video in `Colab_Eiv_TF_HighLevelAPI_3Layer_NN/` | [README](Colab_Eiv_TF_HighLevelAPI_3Layer_NN/README.md) |

---

## 📓 Detailed Notebook Descriptions

### Colab A: NumPy-Only 3-Layer DNN (`Colab_A_NumPy_3Layer_NN.ipynb`)

**Framework:** NumPy (with `tf.einsum` for matrix multiplication)

**Key Features:**
- Pure NumPy implementation — no framework autograd
- **`tf.einsum('ij,jk->ik', A, B)`** used instead of `np.dot` or `@` for all matrix multiplications
- **Manual forward pass** through 3 hidden layers + output
- **Manual backpropagation** implementing chain rule gradient propagation:
  - `dL/dW4 = A3ᵀ @ dZ4`
  - `dL/dW3 = A2ᵀ @ (dZ4 @ W4ᵀ ⊙ ReLU'(Z3))`
  - `dL/dW2 = A1ᵀ @ (... chain continues ...)`
  - `dL/dW1 = Xᵀ @ (... chain continues further ...)`
- He initialization for weights
- Mini-batch gradient descent
- 4D visualization with PCA (scikit-learn)

**Sections:**
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

### Colab B: PyTorch From Scratch (`Colab_B_PyTorch_Scratch_3Layer_NN.ipynb`)

**Framework:** PyTorch (raw tensors only)

**Key Features:**
- **NO `nn.Module`**, NO `nn.Linear`, NO `nn.functional`
- Raw `torch.Tensor` with `requires_grad=True`
- Matrix multiplication via `torch.mm()`
- ReLU implemented as `torch.clamp(Z, min=0)`
- **NO optimizer** — manual `p -= lr * p.grad` updates
- PyTorch autograd computes gradients, but weight updates are manual

**What's NOT used:**  `nn.Module`, `nn.Linear`, `nn.ReLU`, `optim.Adam`, `optim.SGD`, `DataLoader`

---

### Colab C: PyTorch Class-Based (`Colab_C_PyTorch_Classes_3Layer_NN.ipynb`)

**Framework:** PyTorch (full nn.Module)

**Key Features:**
- `ThreeLayerDNN(nn.Module)` class with `__init__` and `forward`
- Uses `nn.Linear`, `nn.ReLU` built-in layers
- Kaiming/He initialization via `nn.init.kaiming_normal_`
- `Adam` optimizer with `ReduceLROnPlateau` scheduler
- `DataLoader` for mini-batch training
- `model.train()` / `model.eval()` mode switching
- `loss.backward()` → `optimizer.step()` training loop

---

### Colab D: PyTorch Lightning (`Colab_D_PyTorch_Lightning_3Layer_NN.ipynb`)

**Framework:** PyTorch Lightning

**Key Features:**
- `ThreeLayerLightningDNN(pl.LightningModule)` — encapsulates model + training
- `NonLinearRegressionDataModule(pl.LightningDataModule)` — encapsulates data pipeline
- `training_step()`, `validation_step()`, `configure_optimizers()` methods
- `pl.Trainer` with callbacks:
  - `EarlyStopping` (patience=100)
  - `ModelCheckpoint` (save best model)
- Automatic logging, progress bars, device management
- `ReduceLROnPlateau` scheduler configured in `configure_optimizers()`

---

### Colab E-i: TensorFlow From Scratch (`Colab_Ei_TF_Scratch_3Layer_NN.ipynb`)

**Framework:** TensorFlow (low-level only)

**Key Features:**
- **NO `tf.keras`** layers — pure `tf.Variable` weights
- `tf.einsum('ij,jk->ik', X, W)` for all matrix multiplications
- `tf.nn.relu()` for activation (NOT keras activation)
- `tf.GradientTape()` for automatic differentiation
- Manual training loop — **NO `model.fit()`**
- `@tf.function` decorator for compiled execution
- `tf.data.Dataset` for batching

---

### Colab E-ii: TensorFlow with Built-in Layers (`Colab_Eii_TF_BuiltInLayers_3Layer_NN.ipynb`)

**Framework:** TensorFlow (keras.layers.Dense + custom loop)

**Key Features:**
- `layers.Dense(64, activation='relu', kernel_initializer='he_normal')` for layer creation
- Layers called explicitly in a `forward()` function
- Still uses `tf.GradientTape()` custom training loop
- **NOT using `model.fit()`** — manual loop with `optimizer.apply_gradients()`
- `tf.keras.losses.MeanSquaredError()` for loss

---

### Colab E-iii: TensorFlow Functional API (`Colab_Eiii_TF_FunctionalAPI_3Layer_NN.ipynb`)

**Framework:** TensorFlow Functional API

**Key Features:**
- `Input(shape=(3,))` → `Dense(...)()` → `Model(inputs, outputs)` pattern
- DAG-based model definition (supports complex topologies)
- `model.compile(optimizer, loss, metrics)` for configuration
- `model.fit()` for training with validation
- Callbacks: `EarlyStopping`, `ReduceLROnPlateau`
- `history` object for loss/metric tracking
- Model visualization with `plot_model()`

---

### Colab E-iv: TensorFlow High-Level API (`Colab_Eiv_TF_HighLevelAPI_3Layer_NN.ipynb`)

**Framework:** TensorFlow/Keras Sequential API

**Key Features:**
- `Sequential([Dense, BatchNorm, Dense, BatchNorm, ...])` — simplest API
- `BatchNormalization` for training stability
- Full `model.compile()` → `model.fit()` → `model.evaluate()` → `model.predict()` pipeline
- Multiple callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`
- `train_test_split` from scikit-learn
- 4D prediction comparison visualization
- Most concise implementation

---

## 🔄 Framework Comparison

| Feature | Colab A | Colab B | Colab C | Colab D | E-i | E-ii | E-iii | E-iv |
|---------|---------|---------|---------|---------|-----|------|-------|------|
| **Framework** | NumPy | PyTorch | PyTorch | PL | TF | TF | TF | TF |
| **Abstraction Level** | Lowest | Low | Medium | High | Lowest | Medium | High | Highest |
| **Manual Backprop** | ✅ | Autograd | Autograd | Autograd | GradTape | GradTape | model.fit | model.fit |
| **Built-in Layers** | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| **Optimizer** | Manual SGD | Manual | Adam | Adam | Adam | Adam | Adam | Adam |
| **tf.einsum** | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **BatchNorm** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Callbacks** | ❌ | ❌ | LR Sched | ES+Ckpt | ❌ | ❌ | ES+LR | ES+LR+Ckpt |

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
1. Upload any `.ipynb` file to [Google Colab](https://colab.research.google.com/)
2. Click **Runtime → Run All**
3. All dependencies are pre-installed on Colab (except `pytorch-lightning` which is installed in Colab D)

### Option 2: Local Jupyter
```bash
pip install numpy tensorflow torch pytorch-lightning matplotlib scikit-learn
jupyter notebook
```

### Option 3: Open from GitHub
Click the "Open in Colab" badge for each notebook (after pushing to GitHub).

---

## 📊 Expected Results

All notebooks should achieve:
- **R² > 0.95** on the test set
- **Loss convergence** visible in training curves
- **Residuals** approximately normally distributed around 0

---

## 🧮 Mathematical Background

### Forward Pass (Layer l)
$$Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}$$
$$A^{[l]} = \text{ReLU}(Z^{[l]}) = \max(0, Z^{[l]})$$

### Backpropagation Chain Rule
$$\frac{\partial L}{\partial W^{[l]}} = (A^{[l-1]})^T \cdot \delta^{[l]}$$

where:
$$\delta^{[L]} = \frac{\partial L}{\partial A^{[L]}} \odot \sigma'(Z^{[L]})$$
$$\delta^{[l]} = (\delta^{[l+1]} \cdot (W^{[l+1]})^T) \odot \sigma'(Z^{[l]})$$

### tf.einsum Usage (Colabs A & E-i)
```python
# Instead of: np.dot(X, W) or X @ W
# We use:
tf.einsum('ij,jk->ik', X, W)  # Matrix multiplication
tf.einsum('ji,jk->ik', A, dZ)  # A^T @ dZ (transpose first matrix)
```

---

## 📦 Dependencies

| Package | Version | Used In |
|---------|---------|---------|
| `numpy` | ≥1.21 | All |
| `tensorflow` | ≥2.10 | A, E-i, E-ii, E-iii, E-iv |
| `torch` | ≥1.12 | B, C, D |
| `pytorch-lightning` | ≥2.0 | D |
| `matplotlib` | ≥3.5 | All |
| `scikit-learn` | ≥1.0 | All (PCA, train_test_split) |

---

## 👤 Author

[Your Name]

## 📄 License

This project is for educational purposes.
