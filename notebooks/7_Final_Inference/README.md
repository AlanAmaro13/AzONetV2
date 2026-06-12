# 7_Final_Inference — Model Evaluation & Results Visualization

Generates comprehensive evaluation figures and statistical reports from the UNET model predictions produced by `6_results_colab`.

## Scripts

| Script | Purpose |
|---|---|
| `1_figures_MSE.py` | Loads predictions, computes error metrics, generates per-split figures and `statistics.txt` |
| `training_plots.py` | Reads `training.log`, plots MSE/MAPE/RMSE training curves over 100 epochs |

## Inputs

| Source | Description |
|---|---|
| `MSE_data/*.npy` | Predictions & ground truth from `6_results_colab` (train/val/test/145) |
| `training.log` | CSV with per-epoch loss, MAPE, RMSE (train + val) |

## Outputs — `MSE_images/`

```
MSE_images/
├── statistics.txt              # Full metrics + distribution analysis for all splits
├── mae.png / mape.png / rmse.png  # Training curves
├── train/                      # 960,000 samples
│   ├── comparsion_true_preds.png    # Scatter true vs pred (colored by APE)
│   ├── dist_ape.png / dist_pe.png   # KDE of APE and PE distributions
│   ├── dist_analysis_ape.png / dist_analysis_pe.png  # Histogram, Q-Q, boxplot, ECDF, moments, summary
│   ├── two_dist.png                 # Dual KDE (true vs predicted thickness)
│   └── error.png                    # APE vs thickness scatter
├── val/                        # 120,000 samples (same figure set)
├── test/                       # 120,000 samples (same figure set)
└── 145/                        # 145 experimental samples (same figure set)
```

## Key Results

| Split | N | MAE (nm) | MSE (nm²) | RMSE (nm) | MAPE (%) | R² |
|---|---|---|---|---|---|---|
| Train | 960,000 | 2.91 | 14.73 | 3.84 | 0.61 | 0.9999 |
| Validation | 120,000 | 2.91 | 14.80 | 3.85 | 0.61 | 0.9999 |
| Test | 120,000 | 2.93 | 15.14 | 3.89 | 0.62 | 0.9999 |
| 145 nm (exp.) | 145 | 58.77 | 7,029.82 | 83.84 | 7.38 | 0.9385 |

AP error distributions are highly skewed and leptokurtic across all simulated splits (skewness 6–7, excess kurtosis 105–167). PE for the 145 nm experimental set is approximately normal (Shapiro-Wilk p = 0.82).

## Dependencies

- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scipy` (stats)
- `scikit-learn` (metrics)

## Usage

```bash
python training_plots.py   # generates mae.png, mape.png, rmse.png
python 1_figures_MSE.py    # generates all per-split figures and statistics.txt
```

All outputs are written to `MSE_images/`.
