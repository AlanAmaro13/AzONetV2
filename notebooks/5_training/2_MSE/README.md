# 5_training — Production Training Runs

This folder contains two final training runs of the U-NET architecture with the best hyperparameters from `4_hp_search`, trained on the full 1.2M synthetic transmittance dataset. The key distinction is the loss function: **MAE vs. MSE**.

## Training Runs

### 1_firstVersion — MAE Loss

**File:** `0_UNET_F145F2.py` / `.ipynb`

The **first production run**, using the AmaroXI custom library with **MAE loss**, consistent with the HP search objective. This serves as the baseline for comparison.

| Aspect | Value |
|--------|-------|
| Loss function | **MAE** (Mean Absolute Error) |
| Optimizer | Adam (lr=0.001) |
| Data source | `0_data_simulation/F145F` |
| Library | AmaroXI (imported, not inline) |
| Architecture | Same as 2_MSE (F=21, nodes [470,340,330], D=35%, L1/L2=1e-6/1e-5, L1C/L2C=1e-6/1e-4) |
| Epochs | 100 |
| Batch size | 512 |
| Seed | 13 |

**Training Results:**

| Metric | Best Val | Epoch | Final Val (Epoch 99) |
|--------|----------|-------|----------------------|
| MAE | **5.76 nm** | 17 | 18.30 nm |
| MAPE | **1.13%** | 17 | 3.99% |
| RMSE | 7.13 nm | 17 | 32.36 nm |

**Observations:**
- Training converges quickly: MAE drops from 226 nm to ~21 nm by epoch 15.
- Best validation at epoch 17 (1.13% MAPE), after which overfitting is moderate but persistent.
- Validation shows high variance epoch-to-epoch (MAE oscillates 5–107 nm), similar to the MSE run.
- Final training MAE: 21.2 nm (5.65% MAPE).
- Model saved as `models/UNET-F145F2/model.h5` and `.keras`.

### 2_MSE — MSE Loss

**File:** `0_UNET_F145F2_Colab_Cleaned.py` / `.ipynb`

The **second production run**, switching the loss to **MSE**. Architecture is identical but the code is self-contained (no AmaroXI dependency, all U-NET/DNN functions defined inline).

Key differences from first version are summarized below.

## Comparison: MAE vs. MSE

| Aspect | 1_firstVersion (MAE) | 2_MSE (MSE) |
|--------|---------------------|-------------|
| Loss function | MAE | **MSE** |
| Library | AmaroXI (imported) | Inline (self-contained) |
| Data path | `0_data_simulation/F145F` | `3_data_simulation/NewDataAzONetV2/Sets` |
| Best Val MAPE | 1.13% (epoch 17) | **0.61%** (epoch 39) |
| Best Val RMSE | 7.13 nm (epoch 17) | **3.85 nm** (epoch 39) |
| Best Val MAE | 5.76 nm (epoch 17) | — (MSE loss, no MAE metric) |
| Overfitting onset | ~epoch 20 | ~epoch 50 |


## Script

**File:** `0_UNET_F145F2_Colab_Cleaned.py` / `0_UNET_F145F2_Colab_Cleaned.ipynb`

A Google Colab script that loads the full dataset, defines the U-NET + DNN architecture with the best hyperparameters from Phase 2 of the HP search, trains for 100 epochs, and generates evaluation plots.

## Architecture

The model combines a 1D U-NET encoder-decoder with a Dense regression head:

**U-NET Section:**

| Component | Value |
|-----------|-------|
| Base filters | 21 (doubles per block: 21 → 42 → 84) |
| Convolutional blocks | 3 (encoder) + 1 (bottleneck) + 3 (decoder) |
| Kernel sizes | [150, 50, 5] |
| Pooling | Average Pooling, pool=4, stride=4 |
| Padding | `valid` |
| Internal activation | LeakyReLU |
| Weight init | He Normal |
| Kernel regularizer | L1=1e-6, L2=1e-4 |
| Output activation | LeakyReLU |
| Output dimension | Matches input (911, 1) after upsampling |

**DNN Head:**

| Component | Value |
|-----------|-------|
| Hidden layers | 3 |
| Nodes | [470, 340, 330] |
| Dropout | 35% |
| L1/L2 regularizers | L1=1e-6, L2=1e-5 |
| Internal activation | LeakyReLU, He Normal |
| Output | 1 neuron, ReLU |

## Data

- **Source:** HDF5 files from `3_data_simulation/NewDataAzONetV2/Sets/` (train.h5, test.h5, val.h5)
- **Full dataset:** ~960k train / ~120k test / ~120k val samples
- **Input:** Transmittance spectra (911, 1), per-sample normalized to [0, 1]
- **Target:** Film thickness (nm)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss function | **MSE** (Mean Squared Error) |
| Metrics | MAPE, RMSE |
| Optimizer | Adam (lr=0.001) |
| Batch size | 512 |
| Epochs | 100 |
| Random seed | 13 |
| Early Stopping | Monitoring `val_mape` (min), patience=1000 (disabled) |
| LR Reduction | Factor 0.8, patience=1000, min_lr=1e-6 (disabled) |
| Checkpoint | Best `val_mape` weights saved as `.keras` |

## Key Difference from HP Search

The HP search in `4_hp_search` used **MAE loss** with **MAPE** as the optimization objective. This production run switches to **MSE loss** — making the training consistent with the physical model fitting in earlier stages (where both DE optimization and data simulation minimized MSE). The MAPE metric is still tracked but the gradient descent now follows the MSE surface.

## Results

The model was trained for 100 epochs. The best validation performance occurred at **epoch 39**, after which overfitting set in (training loss continued to improve while validation oscillated upward).

### Best Model (Epoch 39, saved via checkpoint)

| Metric | Train | Validation |
|--------|-------|------------|
| MSE | — | **18.28 nm²** |
| MAPE | — | **0.61%** |
| RMSE | — | **3.85 nm** |

### Final Model (Epoch 99)

| Metric | Train | Validation |
|--------|-------|------------|
| MSE | 660.20 nm² | 260.79 nm² |
| MAPE | 4.60% | 2.20% |
| RMSE | 25.58 nm | 15.96 nm |

### Training Progress Overview

| Epoch | Train MSE | Train MAPE | Val MSE | Val MAPE | Notes |
|-------|-----------|------------|---------|----------|-------|
| 0 | 178,780 | 48.70% | 20,029 | 20.62% | Initial |
| 1 | 2,718 | 9.95% | 862 | 6.51% | Rapid drop |
| 12 | 799 | 5.18% | 1,634 | 4.52% | |
| 39 | — | — | 18.28 | **0.61%** | **Best (saved)** |
| 50 | 689 | 4.69% | 23.3 | 0.79% | |
| 80 | 669 | 4.62% | 1,784 | 4.11% | Overfitting visible |
| 99 | 660 | 4.60% | 261 | 2.20% | Final |

## Output Files

All outputs are saved to `models/UNET-F145F2-New/`:

| File | Description |
|------|-------------|
| `UNET-F145F2-New.keras` | Best model weights (epoch 39, best val_mape) |
| `model.keras` | Final model weights (epoch 99) |
| `training.log` | CSV log of all 100 epochs (loss, mape, rmse, val_* per epoch) |
| `model.png` | Model architecture diagram |
| `mae.png` | Training vs validation MSE over epochs |
| `mape.png` | Training vs validation MAPE over epochs |
| `rmse.png` | Training vs validation RMSE over epochs |
| `logs/` | TensorBoard event files for train and validation |

## Conclusion

The switch from MAE to MSE loss improves validation MAPE from **1.13% to 0.61%** — nearly a 2× reduction. MSE penalizes large errors more aggressively, which appears to benefit thickness prediction from transmittance spectra. However, both loss functions show overfitting after peaking, and both exhibit high validation variance, suggesting that further regularization or data augmentation may be needed.

## All Output Files

### 1_firstVersion
`models/UNET-F145F2/`: model.h5, UNET-F145F2.keras, training.log, mae.png, mape.png, rmse.png, model.png, logs/

### 2_MSE
`models/UNET-F145F2-New/`: UNET-F145F2-New.keras, model.keras, training.log, mae.png, mape.png, rmse.png, model.png, logs/

## Dependencies

- `tensorflow`, `keras`
- `h5py`, `numpy`, `pandas`
- `matplotlib`, `seaborn`
- `google.colab` (Colab-specific drive mount)
