# 5_training/2_MSE — U-NET Production Training with MSE Loss

This folder contains the final production training of the best U-NET architecture (identified in `4_hp_search`) on the full 1.2M synthetic transmittance dataset. Unlike the hyperparameter search, this run uses **MSE loss** and trains for 100 epochs on the complete dataset.

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

## Dependencies

- `tensorflow`, `keras`
- `h5py`, `numpy`, `pandas`
- `matplotlib`, `seaborn`
- `google.colab` (Colab-specific drive mount)
