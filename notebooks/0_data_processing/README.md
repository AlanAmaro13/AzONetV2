# 0_data_processing

Pipeline to construct the experimental dataset by merging transmittance spectra (`.txt`) with thickness measurements (`.csv`), correcting thickness values where needed, and selecting the best-quality samples.

## Notebooks

### 0 — 101 Samples

**File:** `0_DataFrame_Muestras_Laboratorio.ipynb`  
**Input:**
- Thickness: `../../experimental_samples/Processed_Data/Thickness-Order-100-1600-3.csv`
- Spectra: `../../experimental_samples/Spectrum_Order/*.txt`

**Process:** Reads 101 samples. Thickness values are **already corrected** — no linear transformation is applied. Generates histograms, KDE, and error distribution plots. Merges thickness + spectra on `Nombre-Arc`.

**Output:** `../../results/dataframe_spectrum_thickness_101.pkl`

---

### 1 — 15 Samples

**File:** `1_DataFrame_Muestras_Laboratorio_15.ipynb`  
**Input:**
- Thickness: `../../experimental_samples/Processed_Data/Example_Sample_IA_2_F_30NEW.csv`
- Spectra: `../../experimental_samples/Muestra_ANZO67-ANZO69/*.txt`

**Process:** Same structure as Notebook 0, but thickness requires a **linear correction**:

$$E_{corrected} = \frac{E_{raw} - 17.897827}{1.1521356}$$

**Output:** `../../results/dataframe_spectrum_thickness_15.pkl`

---

### 2 — 45 Samples

**File:** `2_DataFrame_Muestras_Laboratorio_45.ipynb`  
**Input:**
- Thickness: `../../experimental_samples/Processed_Data/Sample_Test_IA_1_New.csv`
- Spectra: `../../experimental_samples/Sample_test/*.txt`

**Process:** Nearly identical to Notebook 1. Applies the **same linear correction** as Notebook 1.

**Output:** `../../results/dataframe_spectrum_thickness_45.pkl`

---

### 3 — Unify All Samples

**File:** `3_Unified_101_45_15.ipynb`  
**Input:** The three `.pkl` files from Notebooks 0, 1, and 2.

**Process:** Concatenates all dataframes into a single dataframe of **161 samples** (101 + 45 + 15). Adds a `Nombre` column to the 45-sample and 15-sample dataframes for schema consistency.

**Output:** `../../results/dataframe_spectrum_thickness_161.pkl`

---

### 4 — Figures

**File:** `4_Figures.ipynb`  
**Input:**
- `../../results/dataframe_spectrum_thickness_161.pkl`
- Precomputed index array: `./indexes/outlier_indices_16.npy` (16 outlier indices)

**Process:** Loads the 161-sample dataframe and drops the 16 outliers identified during manual inspection, yielding the final **145 samples**. Computes the mean percentage error (~9.61%). Generates and saves 5 figures to `./images/`: thickness histograms at 200 nm, 100 nm, and 50 nm bins, a percentage error histogram (1% bins), and a KDE density plot.

**Output:**
- `../../results/dataframe_spectrum_thickness_145_final.pkl` (the final, cleaned dataset)
- `./images/bin_200.png`, `bin_100.png`, `bin_50.png`, `error1.png`, `kde.png`

---

## Key Differences

| Notebook | Samples | Thickness Correction | Spectra Source |
|----------|---------|---------------------|----------------|
| 0        | 101     | None (pre-corrected) | `Spectrum_Order/` |
| 1        | 15      | Linear: `(x - 17.90) / 1.15` | `Muestra_ANZO67-ANZO69/` |
| 2        | 45      | Linear: `(x - 17.90) / 1.15` | `Sample_test/` |

- Notebooks 1 and 2 came from a different measurement batch, requiring the same thickness calibration.
- Notebooks 0–2 produce independent dataframes; Notebook 3 merges them.
- Notebook 4 drops 16 manually-identified outlier samples in a single step (161 → 145), then generates figures and saves them to `./images/`.

## Result

The final dataset (`dataframe_spectrum_thickness_145_final.pkl`) contains 145 samples, each with:
- **Nombre** & **Nombre-Arc**: sample identifiers.
- **Espesor**: corrected thickness (nm).
- **Error** & **Error Porcentual**: thickness measurement error and relative error.
- **Espectro**: transmittance spectrum (911 wavelength points).
