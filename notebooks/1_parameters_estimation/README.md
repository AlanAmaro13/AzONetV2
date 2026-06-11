# Parameters Estimation — Alonso's Transmittance Model Fitting

This folder contains the fitting of experimental transmittance spectra to the theoretical model proposed by Alonso et al. (IIM — Interference Induced Method) using three global optimization algorithms from SciPy.

## Context

In the previous folder (`0_data_processing`), a DataFrame was created containing experimental spectra and their thicknesses. The goal here is to fit each experimental transmittance curve with the theoretical model by minimizing the Mean Square Error (MSE).

## Transmittance Model

The Alonso model describes transmittance as a function of **13 parameters**:

| Parameter | Description |
|-----------|-------------|
| `x` | Wavelength [nm] |
| `d` | Film thickness [nm] |
| `R1`, `R2` | Surface roughness parameters |
| `A`, `B`, `C`, `D`, `E` | Sellmeier coefficients (background refractive index) |
| `alpha`, `beta`, `gamma` | Absorption coefficients |
| `ne` | Free carrier concentration |

The model incorporates:
- **Sellmeier equation** for the background dielectric constant
- **Drude model** for free-carrier contribution (depends on `ne`)
- **Absorption model** with Urbach tail
- **Roughness corrections** for interfaces (via scalar scattering theory)
- **Glass substrate** transmittance (`TexpglassO.txt`) to compute `ng`

The refractive index is constrained to the **physically meaningful range**: `1.8 ≤ n(λ) ≤ 2.1`.

## Data

- **Source**: `../../results/dataframe_spectrum_thickness_145_final.pkl`
- **Samples**: 145 experimental thin-film samples (AZO — Aluminium-doped Zinc Oxide)
- **Spectrum range**: 190–1100 nm (911 points per spectrum)
- **Glass transmittance**: `../../experimental_samples/Background_data/TexpglassO.txt`

## Optimization Algorithms

Three global optimization algorithms were tested:

### 1. Basin-Hopping (`basin_hopping`)
- **File**: `1_SciPy_AllSamples_BH_NonLinear_Porcentual_145F.ipynb`
- **Local optimizer**: SLSQP
- **Constraint**: `NonlinearConstraint` on refractive index
- **Initial guess**: Mean values from prior EGP-GA results
- **Bounds**: Allowed ±50% variation around seed values, ± measurement error for thickness

### 2. Differential Evolution (`differential_evolution`)
- **File**: `2_SciPy_AllSamples_DE_NonLinear_Porcentual_145F.ipynb`
- **Constraint**: `NonlinearConstraint` on refractive index
- **Parallelization**: `workers=-1` (uses all available CPUs)
- **Stochastic**: Random seed per run

### 3. Direct (`direct`)
- **File**: `3_SciPy_AllSamples_Direct_NonLinear_Porcentual_145F.ipynb`
- **Constraint**: Penalty method (Direct does not natively support `NonlinearConstraint`)
- **Fastest** but weakest results

## Results

Results are saved as NumPy arrays in `../../results/SciPy_IIM/`:

| File | Algorithm |
|------|-----------|
| `basin_hopping_NonLinear_145F.npy` | Basin-Hopping |
| `differential_evolution_NonLinear_145F.npy` | Differential Evolution |
| `direct_NonLinear_145F.npy` | Direct |

Each array has shape `(145, 14)` with columns:

```
[SampleIndex, Thickness, R1, R2, A, B, C, D, E, alpha, beta, gamma, ne, MSE]
```

## Comparison & Conclusions

The notebook `4_Comparison.ipynb` (which also generates `comparativa_optimizadores.pdf`) compares the three optimizers:

| Optimizer | Mean MSE | Execution Time | Quality |
|-----------|----------|----------------|---------|
| **Differential Evolution** | ~0 (best) | ~4 hours | Excellent |
| **Basin-Hopping** | ~5 | ~12 hours | Moderate |
| **Direct** | ~8–30 | ~5 minutes | Poor |

**Key findings**:

- **Differential Evolution** is the best optimizer for this problem — it achieves the lowest MSE consistently across all samples. Its KDE density distribution is tightly centered near zero.
- **Basin-Hopping** produces a normal-like distribution centered around MSE ≈ 5, with significantly longer runtime.
- **Direct** is fast but produces large errors, with a multi-modal error distribution. Not recommended for this problem.

The comparison visualizations are saved in the `images/` folder:
- `comparative.png` — Scatter plot of MSE per sample for all three algorithms
- `kde_optimizers.png` — Kernel Density Estimation of the MSE distributions
- `final_comparative.png` — Bar chart of mean MSE comparison

## Test-400nm-1100nm Subfolder

This subfolder explores the effect of **restricting the fitting range** from the full spectrum (190–1100 nm) to the visible–NIR region (400–1100 nm). The idea is that the low-wavelength UV tail (190–400 nm) is noisy and dominated by absorption, so excluding it might improve the fit for the remaining spectral range.

### Adjustment Notebook

**File:** `Test-400nm-1100nm/Adjustment_400nm_1100nm.ipynb`

Reruns **Differential Evolution** fitting on all 145 samples, but the MSE error function only considers wavelengths ≥400 nm (`y_pred[210:]` in the 911-point array). The transmittance model, constraints (`NonlinearConstraint` on refractive index 1.8 ≤ n ≤ 2.1), bounds (±50% around seeds, ± measurement error for thickness), and glass substrate data (`TexpglassO.txt`) are identical to the main DE notebook. All 145 samples are processed (not just 1). Results are saved as `differential_evolution_NonLinear_145F.npy` in the subfolder.

### Comparison Notebook

**File:** `Test-400nm-1100nm/Compare_Old_New_Fittings.ipynb`

Compares the two fitting strategies across all 145 spectra:

| Metric | Old (190–1100 nm) | New (400–1100 nm) |
|--------|------------------|-------------------|
| Mean MSE | 1.099 | 1.186 |
| Median MSE | 0.757 | 0.743 |
| Wins (lower MSE) | **96 / 145 (66.2%)** | 49 / 145 (33.8%) |

**Key findings:**

- The **old (full 190–1100 nm) fitting** wins more often, with a lower mean MSE. The UV region provides additional constraints that help the optimizer converge.
- When the new fit is better, the average improvement is −0.256 MSE; when the old fit is better, the average regression is +0.262 MSE — roughly symmetric.
- The comparison notebook also generates detailed per-sample scatter plots, MSE difference bar charts, ranked top-10 improved/regressed tables, and a grid of 145 experimental vs. model spectrum plots (`images/`).

**Data files in subfolder:**
- `old_data.npy` — (145, 14) parameters + MSE from full 190–1100 nm fitting
- `new_data.npy` — (145, 14) parameters + MSE from 400–1100 nm fitting
- `images/` — generated comparison plots

**Takeaway:** Fitting the full 190–1100 nm range is recommended for production use. The 400–1100 nm restricted fit can serve as a sensitivity analysis but does not improve overall fitting quality.

## Python_Notebooks Subfolder

Contains `.py` script versions of the three optimization notebooks, refactored to use the **AmaroX** custom library instead of inline model definitions:

| Script | Equivalent Notebook |
|--------|-------------------|
| `Python_Notebooks/1_SciPy_AllSamples_BH_NonLinear_Porcentual_145F.py` | Notebook 1 (Basin-Hopping) |
| `Python_Notebooks/2_SciPy_AllSamples_DE_NonLinear_Porcentual_145F.py` | Notebook 2 (Differential Evolution) |
| `Python_Notebooks/3_SciPy_AllSamples_Direct_NonLinear_Porcentual_145F.py` | Notebook 3 (Direct) |

These scripts import the `modelo_transmitancia` function and other utilities from `AmaroX` rather than defining them inline, making them cleaner and more maintainable for production use. The model equations, constraints, bounds, and output paths are identical to their notebook counterparts.

## Generated Report

The notebook `4_Comparison.ipynb` uses the `fpdf` library to produce a PDF report saved as `comparativa_optimizadores.pdf` in this folder. It includes the comparative scatter plot, KDE density comparison, and bar chart with Spanish-language commentary on the optimization results.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `1_SciPy_AllSamples_BH_NonLinear_Porcentual_145F.ipynb` | Basin-Hopping fitting (interpreted) |
| `2_SciPy_AllSamples_DE_NonLinear_Porcentual_145F.ipynb` | Differential Evolution fitting (includes outputs: plots, params, timing) |
| `3_SciPy_AllSamples_Direct_NonLinear_Porcentual_145F.ipynb` | Direct fitting (includes outputs: plots, params, timing) |
| `4_Comparison.ipynb` | Results comparison, KDE, bar charts, and PDF report generation (`comparativa_optimizadores.pdf`) |
| `Python_Notebooks/*.py` | Python script versions using the `AmaroX` library (same logic, alternative imports) |

## Dependencies

- `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`
- `fpdf` (for PDF report generation in `4_Comparison.ipynb`) 
- `AmaroX` (custom library, used only in `Python_Notebooks/*.py`)
