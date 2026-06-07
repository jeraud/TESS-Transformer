# ASTRAFier

A transformer-based classifier for TESS light curves.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Predict with Pre-trained Model

Classify your FITS files using our pre-trained weights:

```bash
astrafier predict --fits-dir /path/to/fits --output-dir ./predictions
```

Output: `predictions.csv` with columns `tic` and `probabilities`.

### Train on Our Data

Download the paper's training data from HuggingFace (includes TESS and Kepler light curves):

```python
from huggingface_hub import hf_hub_download

for f in ["train.safetensors", "train.json", "test.safetensors", "test.json"]:
    hf_hub_download(
        repo_id="paulg9/astrafier_dataset",
        filename=f,
        repo_type="dataset",
        local_dir="./data",
    )
```

Fine-tune from our checkpoint:

```bash
astrafier train --train-pt data/train.safetensors --test-pt data/test.safetensors
```

Or train from scratch:

```bash
astrafier train --train-pt data/train.safetensors --test-pt data/test.safetensors --no-load-from-hf
```

### Train on Your Data

**From CSV** — splitting and preprocessing are handled automatically:

```bash
astrafier train --train-csv data.csv
```

Your CSV needs `label` and `path` columns. If you include a `TIC` column, we split by TIC to prevent data leakage (same star in train and test). Without it, we split by row.

| label | path | TIC |
|-------|------|-----|
| ECLIPSE | /data/tic123.fits | 123 |
| CONTACT_ROT | /data/tic456.fits | 456 |

Save preprocessed tensors for faster reruns:

```bash
astrafier train --train-csv data.csv --save-preprocessed ./tensors
```

**From tensors** — if you have your own preprocessing:

```bash
astrafier train --train-pt train.pt --test-pt test.pt
```

Expected `.pt` structure:

```python
torch.save({
    "flux": flux_tensor,      # (N, seq_len) float32
    "time": time_tensor,      # (N, seq_len) float32
    "labels": label_tensor,   # (N, num_classes) float32 one-hot
    "mask": mask_tensor,      # (N, seq_len) bool, True = valid
    "label_map": {"ECLIPSE": 0, ...},
}, "train.pt")
```

### Predict with Your Checkpoint

```bash
astrafier predict --fits-dir /path/to/fits --checkpoint model.ckpt --no-load-from-hf
```

## Training Options

**Data source**:
- `--train-csv`: From CSV with FITS paths (we split + preprocess)
- `--train-pt` / `--test-pt`: From pre-processed tensors

**Model weights**:
- `--load-from-hf`: Fine-tune from our checkpoint (default)
- `--no-load-from-hf`: Train from scratch

**Other flags:**
- `--batch-size`: Default 128
- `--max-epochs`: Default 250
- `--precision`: Use `32-true` if bf16 not supported
- `--save-preprocessed`: Save tensors when using `--train-csv`
- `--mc-dropout` / `--mc-samples`: Monte Carlo dropout for uncertainty

Run `astrafier train --help` for all options.

## Class Labels

The model classifies into 8 categories:
- `APERIODIC`
- `CONTACT_ROT`
- `DSCT_BCEP`
- `ECLIPSE`
- `GDOR_SPB`
- `INSTRUMENT/JUNK`
- `RRLYR_CEPH`
- `SOLARLIKE`

## Preprocessing

Our TESS preprocessing pipeline (used by `--train-csv` and `astrafier preprocess`):

1. Filter by QUALITY flags (keep 0, 64, 256, 1024, 2048, 8192)
2. Remove NaNs and 10σ outliers
3. Subtract Gaussian-filtered trend (σ=61)
4. Standardize (median=0, std=1)
5. Pad/truncate to sequence length (default: 1171)

## Building Blocks

For more control, use the individual commands:

```bash
# Split CSV by TIC (prevents data leakage)
astrafier split --csv data.csv --output-dir ./splits

# Preprocess to tensors
astrafier preprocess --csv splits/train.csv --output train.pt
astrafier preprocess --csv splits/test.csv --output test.pt

# Train
astrafier train --train-pt train.pt --test-pt test.pt
```

## Project Layout

```
src/astrafier/
├── commands/       # CLI: train, predict, preprocess, split
├── data/
│   └── loading.py  # FITS preprocessing
└── models/         # ASTRAFier architecture
```

## Testing

```bash
pytest
```

## Data and Model
- **Training data:** [Hugging Face dataset](https://huggingface.co/datasets/paulg9/astrafier_dataset) — DOI [10.57967/hf/9082](https://doi.org/10.57967/hf/9082)
- **Trained model:** [Hugging Face model](https://huggingface.co/paulg9/astrafier)