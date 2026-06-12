# 3_data_simulation — Synthetic Dataset Generation via GMM

This folder synthesizes a large-scale dataset of transmittance spectra using **Gaussian Mixture Models (GMM)** fitted to the Differential Evolution parameter distributions from `2_de/bins/`. The pipeline produces ~1.2M simulated thin-film transmittance spectra split into Train/Test/Val for machine learning.

## Pipeline Overview

```
2_de/bins/region_{0..4}.npy          ── DE-fitted parameters per region (13 params each)
         │
         ▼
1_*_Bin_{0..4}_Colab.py  (×5)        ── Phase 1: GMM fit → 3000 synthetic values per param
         │                              Phase 2: Random sampling → 240k spectra per bin → Parquet
         ▼
data/R{1..5}/*.npy                    ── GMM parameter pools (3000 values × 13 params × 5 regions)
NewDataAzONetV2/bin_{0..4}.parquet    ── ~240k rows each (spectra + all parameters)
         │
         ▼
2_Figures_Colab.py                    ── Ridge plots: original DE (ρ) vs simulated (ρₛ) KDEs
         │                              Spectrum samples, thickness histograms
         ▼
images/                               ── Comparison visualizations
         │
         ▼
3_join_split_merge_Colab.py           ── Merge 5 parquets → ~1.2M rows → shuffle
         │                              80% train / 10% test / 10% val → HDF5
         ▼
NewDataAzONetV2/Final_Dataset.parquet ── Merged + shuffled
NewDataAzONetV2/Sets/
  ├── train.h5  (80%  · ~960k samples)
  ├── test.h5   (10%  · ~120k samples)
  └── val.h5    (10%  · ~120k samples)
```

## Source Files

| File | Description |
|------|-------------|
| `Python_Notebooks/1_*_Bin_0_Colab.py` | Simulate **Region R₁** (100–250 nm) |
| `Python_Notebooks/1_*_Bin_1_Colab.py` | Simulate **Region R₂** (250–600 nm) |
| `Python_Notebooks/1_*_Bin_2_Colab.py` | Simulate **Region R₃** (600–950 nm) |
| `Python_Notebooks/1_*_Bin_3_Colab.py` | Simulate **Region R₄** (950–1100 nm) |
| `Python_Notebooks/1_*_Bin_4_Colab.py` | Simulate **Region R₅** (1100–1500 nm) |
| `Python_Notebooks/2_Figures_Colab.py` | Ridge plots, spectrum samples, thickness comparison |
| `Python_Notebooks/3_join_split_merge_Colab.py` | Merge bins, shuffle, Train/Test/Val split |
| `1_*_Bin_*_Colab.ipynb` (×5) | Jupyter notebook versions of the bin scripts |
| `2_Figures_Colab.ipynb` | Jupyter notebook version of figures |
| `3_join_split_merge_Colab.ipynb` | Jupyter notebook version of merge/split |

## Detailed Process

### Phase 1 — Gaussian Mixture Model Fitting

Each bin script loads its region's DE-fitted parameters from `../../2_de/bins/region_{N}.npy` (a `(n_samples, 13)` array excluding MSE). For each of the **13 parameters**, the script:

1. Fits a **Gaussian Mixture Model** with up to 3 components, selecting the optimal number via **BIC** (Bayesian Information Criterion).
2. Extracts means (`μᵢ`), standard deviations (`δᵢ`), and mixing weights (`wᵢ`).
3. Generates **3000 synthetic values** using `np.random.normal(μᵢ, δᵢ, n·wᵢ)`.
4. For thickness specifically, values are **clamped** to the region's bounds (e.g., `[100, 250]` for bin 0).
5. For other parameters, values are clamped to `[min(data), max(data)]`.
6. GMM parameters (centroids, std devs, weights, FWHM) are printed and formatted into a PrettyTable with LaTeX export.

**Sellmeier constraint**: The 5 Sellmeier coefficients (A, B, C, D, E) are **not independent** — they must produce a refractive index within `[1.8, 2.1]` across 190–1100 nm. The script generates 3000 valid (A,B,C,D,E) tuples by rejection sampling:

```python
while len(ABCDE) < 3000:
    A,B,C,D,E = pick_one(pool_A), pick_one(pool_B), ..., pick_one(pool_E)
    n_min, n_max = refractive_index(A,B,C,D,E)  # across 190-1100 nm
    if 1.8 <= n_min and n_max <= 2.1:
        ABCDE.append((A,B,C,D,E))
```

The individual A, B, C, D, E pools are then overwritten with the constrained tuples.

### Phase 2 — Transmittance Spectrum Generation

With 3000 parameter values available for each of the 13 parameters in a bin, the script generates **240,000 synthetic transmittance spectra** by:

1. **Randomly sampling** one value from each parameter's pool using `np.random.choice` (parameters are sampled independently).
2. **Computing** the transmittance spectrum via the full Alonso model (`modelo_transmitancia`) at 911 wavelength points (190–1100 nm).
3. **Storing** each spectrum along with its Espesor, R1, R2, Sellmeier coefficients, Absorption coefficients, and ne.
4. **Shuffling** the final DataFrame and saving as Parquet.

### Phase 3 — Figures

`2_Figures_Colab.py` generates comparison visualizations between the **original DE-fitted distributions** and the **simulated GMM distributions**:

- **Ridge plots** for all 12 parameters across 5 thickness regions, with overlaid KDEs:
  - Blue (`ρ`): original DE-fitted parameter distributions
  - Red (`ρₛ`): simulated GMM distributions
- **Refractive index** ridge plot comparing original vs. simulated n(λ) per region.
- **Spectrum samples**: representative transmittance curves from regions R₁, R₃, and R₅ with thickness annotations.
- **Thickness histograms**: 50 nm bins for both the combined simulated thickness (all 1.2M samples) and the original 145 experimental samples, with region dividers.

### Phase 4 — Merge & Split

`3_join_split_merge_Colab.py`:

1. **Loads** the 5 Parquet files (`bin_{0..4}.parquet`).
2. **Concatenates** into a single DataFrame (~1.2M rows).
3. **Shuffles** with `sample(frac=1)`.
4. **Saves** `Final_Dataset.parquet`.
5. **Extracts** `Espectro` → reshaped to `(n, 911, 1)` and `Espesor` → `(n, 1)`.
6. **Splits** into Train/Test/Val at **80% / 10% / 10%**.
7. **Saves** each split as HDF5 with datasets `x_total` (spectra) and `y_total` (thickness), with chunked storage for extensibility.

## Transmittance Model

The same Alonso model from previous folders is re-implemented inline, including:
- **Sellmeier** equation for background dielectric constant
- **Drude model** for free-carrier effects (ω = 2πc·10⁹/x, γ = 2.8×10¹¹·x, factor 3182.61)
- **Urbach tail** absorption (with constant 1240)
- **Scalar scattering theory** roughness corrections (σ₁ at air/film, σ₂ at film/glass)
- **Multi-layer interference** (film on glass, glass ng from `TexpglassO.txt`)
- **Refractive index constraint**: `1.8 ≤ n(λ) ≤ 2.1`

Constants: c = 3×10⁸ m/s, μ = 3.90×10⁻⁴ m²/Vs.

## Output Directory Structure

```
data/                                    # GMM parameter pools (13 .npy files each, 3000 values)
├── R1/   (thickness.npy, r1.npy, r2.npy, A.npy … ne.npy)
├── R2/
├── R3/
├── R4/
└── R5/
images/
├── ridger_plot/                         # Original (ρ) vs simulated (ρₛ) ridge plots per parameter
│   ├── thickness.png, sigma1.png, sigma2.png
│   ├── A.png, B.png, C.png, D.png, E.png
│   ├── alpha.png, beta.png, lambda.png, ne.png, indice.png
│   ├── rugosidades/  (1thickness.png, sigma1.png, sigma2.png)
│   ├── Absorcion/    (alpha.png, beta.png, lambda.png, ne.png)
│   ├── Sellmeier/    (A.png, B.png, C.png, D.png, E.png, indice.png)
│   └── combined_*.png
├── spectrum_thickness/                  # Sample spectra + thickness histograms
│   ├── r1.png, r3.png, r5.png
│   ├── allthickness.png, 145thickness.png
│   └── combined_spectrums.png
└── combined_hist_thickness.png          # 2-panel grid (all simulated + 145 original)
NewDataAzONetV2/
├── bin_0.parquet                        # ~240k rows (R₁: 100–250 nm)
├── bin_1.parquet                        # ~240k rows (R₂: 250–600 nm)
├── bin_2.parquet                        # ~240k rows (R₃: 600–950 nm)
├── bin_3.parquet                        # ~240k rows (R₄: 950–1100 nm)
├── bin_4.parquet                        # ~240k rows (R₅: 1100–1500 nm)
├── Final_Dataset.parquet                # ~1.2M rows merged + shuffled
└── Sets/
    ├── train.h5   (80%  · ~960k)        # x_total (n,911,1) + y_total (n,1)
    ├── test.h5    (10%  · ~120k)
    └── val.h5     (10%  · ~120k)
```

## Dependencies

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scipy`, `scipy.stats` (gaussian_kde)
- `sklearn.mixture` (GaussianMixture)
- `tqdm` (progress bars)
- `PIL` (Pillow, image grids)
- `prettytable` (GMM parameter tables + LaTeX export)
- `h5py` (HDF5 export for ML datasets)
- `pyarrow` or `fastparquet` (Parquet I/O)
- `google.colab` (Google Drive mount, Colab-specific)
