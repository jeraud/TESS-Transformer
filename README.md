# Astrafier

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```


## Data Preparation

Training relies on a CSV catalogue that maps class labels to FITS files. The minimal schema is:

| label | path |
|-------|------|
| ECLIPSE | data/sector14/curve1.fits |

- `label`: class name (one of `APERIODIC`, `CONTACT_ROT`, `DSCT_BCEP`, `ECLIPSE`, `GDOR_SPB`, `JUNK`, `RRLYR_CEPH`, `SOLARLIKE`).
- `path`: absolute or relative path to a FITS file. Relative paths are resolved with respect to the CSV location.
- FITS files follow the following preprocessing pipeline: flagged cadences (QUALITY not in `{0,64,256,1024,2048,8192}`) are dropped, time is zero-based, a 1-D Gaussian trend is removed, and the residual is standardised and padded to the target sequence length (default 1171 for TESS primary mission curves). The generated mask is `True` for real samples and `False` for padded elements.

For inference you only need a directory of FITS files. The CLI traverses the directory (recursively if requested), applies the same preprocessing, and infers TIC IDs from the FITS headers. Files without a TIC ID are assigned incremental identifiers.

## Quick Start

1. **Prepare a CSV catalogue** (at minimum `label,path`). Relative paths are resolved relative to the CSV location.
2. **Train**:
   ```bash
   astrafier train \
     --train-csv data/catalog.csv \
     --output-dir artifacts/train_run \
     --max-epochs 50
   ```
   - HuggingFace weights are loaded by default from https://huggingface.co/paulg9/astrafier; this checkpoint was trained on TESS QLP primary-mission curves. Add `--no-load-from-hf` or `--checkpoint` to start from scratch or from a different checkpoint.
   - Precision defaults to `bf16-mixed` and batch size to 256. Depending on your device, you may need to switch to `--precision 32-true` and shrink the batch size (`--batch-size`).
   - Checkpoints plus `confusion_matrix_<acc>.png` are written to `--output-dir`.
3. **Predict**:
   ```bash
   astrafier predict \
     --checkpoint artifacts/train_run/cl_model_0.95.ckpt \
     --fits-dir data/inference_curves \
     --output-dir artifacts/predictions \
     --no-load-from-hf
   ```
   - Output is `predictions.csv` (`tic`, `probabilities`).
   - Monte Carlo dropout is **off** by default. Add `--mc-dropout --mc-samples 10` to produce better uncertainty-aware predictions.
   - These commands are examples—run `astrafier train --help` or `astrafier predict --help` to see every available flag.


## Testing

Run the unit and integration tests locally with:

```bash
pytest
```


## Project Layout

- `src/astrafier/` – installable package with models, data loaders, and CLI commands.
- `astrafier` console script – entry point providing `astrafier train` and `astrafier predict`.
- `tests/` – unit and integration tests.
- `requirements.txt` – runtime dependencies.
