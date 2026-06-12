# 6_results_colab — UNET Inference & Prediction Pipeline

Generates thickness predictions from spectral data using a pre-trained UNET model, producing train, test, validation, and experimental (145 nm) results saved as `.npy` files.

## Dependencies

- `tensorflow` / `keras`
- `h5py`, `numpy`, `pandas`
- `matplotlib`, `seaborn`
- `google.colab` (drive mount)

## Inputs

| Source | Path | Description |
|---|---|---|
| Train/Test/Val `.h5` | `notebooks/3_data_simulation/NewDataAzONetV2/Sets/` | Normalized spectra + thickness labels |
| UNET model | `notebooks/6_results_colab/models/Copia de model.keras` | Pre-trained Keras model |
| Experimental spectra | `results/dataframe_spectrum_thickness_145_final.pkl` | Experimental 145 nm spectra |
| DE parameters | `results/SciPy_IIM/differential_evolution_NonLinear_145F.npy` | Thickness values from differential evolution |

## Outputs

All outputs are saved under `MSE_data/`:

| File | Content |
|---|---|
| `preds_train.npy` / `y_train.npy` | Train predictions & ground truth |
| `preds_test.npy` / `y_test.npy` | Test predictions & ground truth |
| `preds_val.npy` / `y_val.npy` | Validation predictions & ground truth |
| `preds_145.npy` / `y_145.npy` | Experimental predictions & ground truth |

## Usage

1. Mount Google Drive
2. Ensure all input paths are accessible
3. Run the script: `python Predictions.py`
4. Collect `.npy` files from `MSE_data/`
