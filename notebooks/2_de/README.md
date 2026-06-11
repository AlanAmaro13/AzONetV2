# notebooks/2_de — Differential Evolution Fitting Analysis

This folder contains the analysis of thin-film transmittance fitting results obtained via **Differential Evolution (DE)**, a global optimization algorithm. All generated data is organized under `data_results/`.

## Contents

### Source Files
- **`0_comparison_dist.py`** — Production Python script version. Analyzes 145 AZO (Aluminum-doped Zinc Oxide) thin-film sample fitting results. Loads optimized parameters and experimentally measured spectra, then generates extensive visualizations and statistical reports.
- **`0_comparison_dist.ipynb`** — Notebook version of the same code (for interactive exploration).

### What the code does

1. **Loads DE fitting results** from `../../results/SciPy_IIM/differential_evolution_NonLinear_145F.npy` — a (145, 14) array containing: thickness, roughness σ₁, roughness σ₂, 5 Sellmeier coefficients (A, B, C, D, E), 3 absorption coefficients (α, β, γ), free-electron concentration nₑ, and MSE.

2. **Loads experimental data** from `../../results/dataframe_spectrum_thickness_145_final.pkl` — contains sample names, thickness estimates, errors, and measured spectra (911 wavelength points, 190–1100 nm).

3. **Transmittance model** — Implements the full Alonso physical model with:
   - **Sellmeier equation** for bandgap refractive index contribution (5 parameters: A, B, C, D, E)
   - **Drude model** for free-electron effects (nₑ, mobility μ = 3.90 × 10⁻⁴ m²/Vs)
   - **Absorption** via Urbach tail (α, β, γ, with constant 1240, γ_wavelength = 2.8 × 10¹¹ s⁻¹·nm, 3182.61 factor)
   - **Surface roughness** via scalar scattering theory (σ₁ at air/film interface, σ₂ at film/glass interface)
   - **Multi-layer interference** (film on glass substrate, including glass transmittance from `TexpglassO.txt`)
   - **Refractive index constraint**: 1.8 ≤ n(λ) ≤ 2.1

4. **Statistical Analysis** — For each of the 13 fitted parameters, the `analyze_distribution()` function computes:
   - Basic statistics (mean, median, std, range, IQR)
   - Distribution shape (skewness, kurtosis with interpretation)
   - Normality tests (Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling, Jarque-Bera)
   - Quantile comparison (empirical vs. theoretical normal)
   - Outlier detection via IQR method
   - Entropy and coefficient of variation
   - A 6-panel diagnostic figure (histogram + normal PDF overlay, Q-Q plot, box plot, ECDF vs CDF, moments bar chart, summary statistics table)

5. **Thickness Modeling** — Fits a Gaussian Mixture Model (GMM) to the thickness distribution using BIC to select the optimal number of components (up to 12). Prints the GMM means, covariances, and weights.

6. **Region-Wise Analysis** — Partitions samples into 5 thickness intervals (R₁: 100–250 nm, R₂: 250–600 nm, R₃: 600–950 nm, R₄: 950–1100 nm, R₅: 1100–1500 nm) and computes per-region statistics (min, max, mean, std, median, quartiles, IQR, skewness, kurtosis, outlier count) for all parameters. Results are saved as `.npy` files in `bins/` and rendered as PrettyTable tables with LaTeX export.

## Output Directory Structure (`data_results/`)

```
data_results/
├── comparison/                  # 145 individual PNGs: experimental vs. fitted transmittance spectra
├── comparison_grids/            # 17 grid overviews (grid_0.png … grid_16.png, 3×3 layout, last grid padded with blanks)
├── dist/                        # Histograms and KDEs for MSE and thickness
│   ├── ECM_DE.png               # MSE histogram (1-nm bins)
│   ├── error_kde.png            # MSE KDE density
│   ├── thickness_450.png        # Thickness histogram (450 nm bins)
│   ├── thickness_200.png        # Thickness histogram (200 nm bins)
│   ├── thickness_100.png        # Thickness histogram (100 nm bins)
│   ├── thickness_50.png         # Thickness histogram (50 nm bins)
│   ├── thickness_50_sep.png     # Thickness histogram with region dividers
│   └── thickness_kde.png        # Thickness KDE with ±1σ lines
├── dist_f/                      # Parameter distribution KDE plots
│   ├── R1R2_kde.png             # r, σ₁, σ₂ distributions (1×3 subplots)
│   ├── sellmeier_kde.png        # A, B, C, D, E, n(λ) distributions (2×3 subplots)
│   └── combined_absorption_ne.png # α, β, λ_g, nₑ distributions (2×2 subplots)
├── distribution_analysis/       # Per-parameter statistical reports (13 subfolders)
│   ├── thickness_r/             # analysis_thickness_r.txt + analysis_thickness_r.png
│   ├── roughness_R1/            # analysis_roughness_R1.txt + analysis_roughness_R1.png
│   ├── roughness_R2/            # …
│   ├── sellmeier_A/
│   ├── sellmeier_B/
│   ├── sellmeier_C/
│   ├── sellmeier_D/
│   ├── sellmeier_E/
│   ├── refractive_index_n/
│   ├── absorption_alpha/
│   ├── absorption_beta/
│   ├── absorption_lambda/
│   └── free_carriers_ne/
├── ridge_plot/                  # 13 ridge plots: per-parameter KDEs across 5 thickness regions
│   ├── thickness.png            # also sigma1.png, sigma2.png, A.png, B.png, C.png, D.png,
│   │                           #   E.png, alpha.png, beta.png, lambda.png, ne.png, indice.png
├── ridge_plot_comparison/       # Grid composites of ridge plots
│   ├── combined_rugosidades.png # r, σ₁, σ₂ (1×3)
│   ├── combined_absorcion.png   # α, β, λ, nₑ (2×2)
│   └── combined_sellmeier.png   # A, B, C, D, E, n(λ) (2×3)
├── latex_tables/                # LaTeX tables for each thickness region
│   ├── latex_bin_0.txt          # Region R₁: [100, 250] nm
│   ├── latex_bin_1.txt          # Region R₂: [250, 600] nm
│   ├── latex_bin_2.txt          # Region R₃: [600, 950] nm
│   ├── latex_bin_3.txt          # Region R₄: [950, 1100] nm
│   └── latex_bin_4.txt          # Region R₅: [1100, 1500] nm
└── samples_per_region.txt       # Count of samples in each thickness region
```

## Pre-computed Data
- **`bins/`** — Contains pre-computed `.npy` files: `all_metrics_{0..4}`, `bin_index_{0..4}`, `min_max_mean_std_{0..4}`, `region_{0..4}` for the 5 thickness intervals, enabling region-wise statistical evaluation.

## Key Libraries
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scipy`, `scipy.stats`, `scipy.optimize`
- `sklearn.mixture` (GaussianMixture)
- `PIL` (Pillow, for image grid generation)
- `prettytable` (for formatted tables and LaTeX export)
