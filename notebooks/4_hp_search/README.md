# 4_hp_search — Hyperparameter Search for ML Models

This folder contains hyperparameter searches for three neural network architectures trained on synthetic transmittance spectra to predict AZO thin-film thickness. The search uses `keras_tuner.BayesianOptimization` with the **AmaroX / AmaroXI** custom libraries.

## Rationale

Deep Neural Networks (DNN) trained on flattened spectra achieved a minimum error of ~10.5% MAPE. This relatively high error is attributed to ignoring the **spatial structure** of transmittance spectra — the position of each transmittance value at its corresponding wavelength uniquely defines the curve. Convolutional architectures, particularly autoencoder-like structures such as **U-NET**, are designed to exploit this spatial dependence, motivating the switch from DNN to U-NET.

## Data

All scripts load synthetic transmittance spectra from the `3_data_simulation` pipeline:
- **DNN scripts:** `../../../../data_simulation/F2` (HDF5)
- **U-NET scripts:** `data_simulation/F1` (HDF5)
- **Format:** HDF5 files with `x_total` (n, 911, 1) — normalized 0–1 transmittance spectra — and `y_total` (n, 1) — film thickness
- **Subset:** 1/10 of the full dataset per search (108k train / 10.8k test / 1.2k val)
- **Loss:** MAE, **metric:** MAPE (Mean Absolute Percentage Error)

## Source Files

### 0_UsingDNN/ — Deep Neural Networks

| Script | Hidden Layers | Node Search | Dropout | Reg (L1, L2) |
|--------|:---:|-------------|---------|------|
| `0_DNN_3HL.ipynb` | 3 | (notebook only) | — | — |
| `0_DNN_5HL.py` / `.ipynb` | 5 | 50–500 per layer, step 50 | 0–50, step 2 | Searched |
| `0_DNN_5HL_Reg.py` / `.ipynb` | 5 | Fixed [400,50,300,100,250] | 0–50, step 5 | Searched |
| `0_DNN_7HL.py` / `.ipynb` | 7 | 50–500 per layer, step 50 | 0–50, step 2 | Not searched |
| `0_DNN_7HL_Reg.py` / `.ipynb` | 7 | Fixed [500,300,50,500,50,400,500] | 0–50, step 5 | Searched |
| `0_DNN_10HL.py` / `.ipynb` | 10 | 50–500 per layer, step 50 | Not searched | Not searched |
| `0_DNN_10HL_Reg.py` / `.ipynb` | 10 | Fixed [500,300,100,200,450,300,150,300,300,500] | 0–50, step 5 | Searched |

All DNN models use: Input (911,) → `G_Dense` layers (leaky_relu, He normal init, L1/L2 optional) → output (1, relu). Optimizer: Adam (lr=0.001). Loss: MAE, metric: MAPE.

### 2_UsingUNET/ — U-NET Architecture

| Script | Description |
|--------|-------------|
| `0_UNET_F1_Regularized.py` / `.ipynb` | Second-phase search: fix best architecture from Phase 1, search regularizers |

## U-NET Architecture (Fixed Components)

The U-NET model, designed to capture the spatial structure of transmittance spectra, uses the following fixed components (selected based on physical considerations of the spectral curves):

**Convolutional Section (encoder-decoder, 3 blocks):**

| Component | Value | Rationale |
|-----------|-------|-----------|
| Convolutional blocks | 3 | Sufficient depth without excessive spatial compression |
| Base filters `F_bloque,0` | Searched (Phase 1) / Fixed at 21 (Phase 2) | Doubles per block: `F_i = 2 · F_{i-1}` |
| Kernel sizes | **[150, 50, 5]** | 150 chosen from max-min distance for films 100–250 nm |
| Pooling | Average Pooling, Pool=4, Stride=4 | Valid padding, output dim: (Dim_in − 4)/4 + 1 |
| Internal activation | Leaky ReLU | |
| Weight init | He Normal | |
| Output activation | LeakyReLU | |

**DNN Head (thickness regression):**

| Component | Value |
|-----------|-------|
| Hidden layers | 3 |
| Nodes per layer | Searched (Phase 1) / Fixed at [470, 340, 330] (Phase 2) |
| Dropout | Searched |
| L1/L2 regularizers | Searched (Phase 2 only) |
| Internal activation | Leaky ReLU, He Normal init |
| Output | 1 neuron, ReLU |

**Training Configuration:**
- Input shape: (911, 1) — 1D spectrum treated as a "1D spatial signal"
- Loss: MAE, Metric: MAPE
- Optimizer: Adam (lr=0.001)
- Batch size: 512
- 50 trials × 2 executions each

## Hyperparameter Search Strategy (Two-Phase)

### Phase 1 — Architecture Search (no regularizers)

**File:** `0_UNET_F1_Regularized.py` was adapted (the original architecture-search script is referenced by `document.txt` but not present in this folder).

**Search Space:**
- `F_bloque,0` ∈ [2, 30], step 1
- 3 DNN layers with `n_i` ∈ [50, 500), step 10
- Dropout ∈ [0, 50%), step 5
- No L1/L2 regularizers

**Results** — `models/best_models_unet3m.txt`

| # | Trial | F_UNET | Nodes | Dropout | Accuracy | ~MAPE |
|---|-------|--------|-------|---------|----------|-------|
| 1 | 187 | **21** | [470, 340, 330] | 25% | 93.82% | ~6.18% |
| 2 | 042 | 25 | [190, 450, 380] | 5%  | 92.87% | ~7.13% |
| 3 | 265 | 19 | [420, 290, 270] | 20% | 91.85% | ~8.15% |
| 4 | 093 | 27 | [130, 150, 180] | 10% | 91.67% | ~8.33% |
| 5 | 211 | 11 | [310, 420, 160] | 40% | 91.53% | ~8.47% |
| 6 | 148 | 21 | [380, 470, 330] | 45% | 91.38% | ~8.62% |
| 7 | 076 | 15 | [290, 320, 440] | 10% | 90.87% | ~9.13% |
| 8 | 299 | 27 | [480, 210, 390] | 30% | 90.26% | ~9.74% |
| 9 | 033 | 13 | [90, 460, 270]  | 35% | 89.92% | ~10.08% |
| 10 | 154 | 29 | [70, 380, 490]  | 20% | 89.66% | ~10.34% |

The best candidate (F=21, units=[470,340,330], D=25%) achieves ~6.18% MAPE — a significant improvement over the DNN baseline (~10.5%).

### Phase 2 — Regularizer Search (fixed best architecture)

**File:** `0_UNET_F1_Regularized.py` / `0_UNET_F1_Regularized.ipynb`

Fixes the best architecture from Phase 1 (F=21, units=[470,340,330]) and searches regularizers to further reduce overfitting.

**Search Space:**
- Dropout ∈ [0, 50), step 5
- L1, L2, L1C, L2C ∈ {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0}
  - L1, L2: DNN section regularizers
  - L1C, L2C: Convolutional section regularizers (L1L2 kernel regularizer)

**Results** — `models/DNN_UNET_F1_R/best_models.txt`

| # | Trial | D | L1 | L2 | L1C | L2C | MAPE |
|---|-------|---|------|------|------|------|------|
| 1 | 29 | 35% | 1e-6 | 1e-5 | 1e-6 | 1e-4 | **5.77%** |
| 2 | 44 | 30% | 1e-6 | 1e-6 | 1e-6 | 1e-6 | 6.88% |
| 3 | 20 | 25% | 1e-6 | 1e-6 | 1e-4 | 1e-5 | 6.99% |
| 4 | 37 | 35% | 1e-6 | 1e-5 | 1e-6 | 1e-3 | 7.10% |
| 5 | 14 | 25% | 1e-6 | 1e-5 | 1e-6 | 1e-4 | 7.23% |
| 6 | 41 | 30% | 1e-6 | 1e-6 | 1e-6 | 1e-6 | 7.64% |
| 7 | 38 | 35% | 1e-6 | 1e-5 | 1e-6 | 1e-4 | 7.67% |
| 8 | 21 | 35% | 1e-6 | 1e-5 | 1e-5 | 1e-4 | 7.70% |
| 9 | 28 | 50% | 1e-6 | 1e-6 | 1e-6 | 1e-3 | 7.84% |
| 10 | 18 | 40% | 1e-6 | 1e-6 | 1e-4 | 1e-3 | 8.15% |

The best candidate (D=35%, L1=1e-6, L2=1e-5, L1C=1e-6, L2C=1e-4) achieves **5.77% MAPE** — a further improvement over the Phase 1 result (6.18%).

## Results Summary

| Model | Best MAPE | Key Hyperparameters |
|-------|-----------|---------------------|
| DNN 5HL | 12.33% | Nodes [400,50,300,100,250] |
| DNN 7HL | 10.79% | Nodes [500,300,50,500,50,400,500] |
| DNN 10HL | 10.51% | Nodes [500,300,100,200,450,300,150,300,300,500] |
| **U-NET Phase 1** | **~6.18%** | F=21, units=[470,340,330], D=25% |
| **U-NET Phase 2** | **5.77%** | D=35%, L1=1e-6, L2=1e-5, L1C=1e-6, L2C=1e-4 |

The U-NET architecture **cuts error nearly in half** compared to the best DNN (5.77% vs 10.51%), confirming the hypothesis that convolutional architectures are better suited for the spatial structure of transmittance spectra.

## Results Files

| File | Content |
|------|---------|
| `0_UsingDNN/models/DNN_HP_5/best_models.txt` | DNN 5HL — top 10 trials (best: 12.33% MAPE) |
| `0_UsingDNN/models/DNN_HP_7/best_models.txt` | DNN 7HL — top 10 trials (best: 10.79% MAPE) |
| `0_UsingDNN/models/DNN_HP_10/best_models.txt` | DNN 10HL — top 10 trials (best: 10.51% MAPE) |
| `2_UsingUNET/models/best_models_unet3m.txt` | U-NET Phase 1 — top 10 trials (best: 93.82% accuracy) |
| `2_UsingUNET/models/DNN_UNET_F1_R/best_models.txt` | U-NET Phase 2 — top 10 trials (best: 5.77% MAPE) |

## Dependencies

- `keras`, `keras_tuner`
- `AmaroX` / `AmaroXI` (custom libraries providing `G_Dense`, `G_UNET`, `load_data_normalization_sample_General`, `standard_callbacks`, `get_gpu`, etc.)
- `pandas`, `numpy`, `h5py`
