"""Utilities for loading and preprocessing light curves from text files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import lightkurve as lk
from scipy.ndimage import gaussian_filter1d

QUALITY_FLAGS_KEEP: Tuple[int, ...] = (0, 64, 256, 1024, 2048, 8192)
DEFAULT_SEQ_LEN = 1171


@dataclass
class LoadedCurve:
    time: torch.Tensor
    flux: torch.Tensor
    mask: torch.Tensor
    ticid: int


def _extract_lightcurve(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
    """
    Extract time and flux from a text file.
    
    Expected format: space-separated columns (time, flux, uncertainty)
    The starname/ticid is extracted from the filename (e.g., '005611160.txt' -> 5611160)
    """
    # Extract starname from filename (remove .txt extension and leading zeros)
    stem = path.stem
    try:
        ticid = int(stem.lstrip('0')) if stem.lstrip('0') else 0
    except ValueError:
        ticid = None
    
    # Read the text file (3 columns: time, flux, uncertainty)
    data = np.loadtxt(path, dtype=np.float64)
    
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {path}, got shape {data.shape}")
    
    time = data[:, 0]
    flux = data[:, 1]
    
    # Remove any NaN or infinite values
    valid_mask = np.isfinite(time) & np.isfinite(flux)
    time = time[valid_mask]
    flux = flux[valid_mask]
    
    return time, flux, ticid


def _preprocess_lightcurve(time: np.ndarray, flux: np.ndarray, seq_len: int) -> LoadedCurve:
    lc = lk.LightCurve(time=time, flux=flux)
    lc = lc.remove_nans().remove_outliers(sigma=10)
    if len(lc.time) == 0:
        padded_time = torch.zeros(seq_len, dtype=torch.float32)
        padded_flux = torch.zeros(seq_len, dtype=torch.float32)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        return LoadedCurve(time=padded_time, flux=padded_flux, mask=mask, ticid=-1)

    trimmed = lc[:seq_len]
    time_values = np.array(trimmed.time.value, dtype=np.float32)
    time_values -= time_values[0]

    flux_values = np.array(trimmed.flux.value, dtype=np.float32)
    window = 61 if len(flux_values) >= 61 else max(len(flux_values) // 2, 1)
    flux_values = flux_values - gaussian_filter1d(flux_values, window)
    std = np.std(flux_values)
    if std > 0:
        flux_values = (flux_values - np.median(flux_values)) / std
    else:
        flux_values = flux_values - np.median(flux_values)

    length = len(time_values)
    padded_time = np.zeros(seq_len, dtype=np.float32)
    padded_flux = np.zeros(seq_len, dtype=np.float32)
    mask = np.zeros(seq_len, dtype=bool)

    padded_time[:length] = time_values[:seq_len]
    padded_flux[:length] = flux_values[:seq_len]
    mask[:length] = True

    return LoadedCurve(
        time=torch.from_numpy(padded_time),
        flux=torch.from_numpy(padded_flux),
        mask=torch.from_numpy(mask),
        ticid=-1,
    )


def load_processed_lightcurve(path: Path, seq_len: int = DEFAULT_SEQ_LEN) -> LoadedCurve:
    time, flux, ticid = _extract_lightcurve(path)
    curve = _preprocess_lightcurve(time, flux, seq_len)
    if ticid is not None:
        curve.ticid = ticid
    else:
        curve.ticid = -1
    return curve


def _build_dataset_from_curves(curves: Sequence[LoadedCurve], labels: Optional[torch.Tensor] = None) -> TensorDataset:
    flux_stack = torch.stack([curve.flux for curve in curves])
    time_stack = torch.stack([curve.time for curve in curves])
    mask_stack = torch.stack([curve.mask for curve in curves])

    if labels is not None:
        return TensorDataset(flux_stack, time_stack, labels, mask_stack)

    ticids = torch.tensor([curve.ticid for curve in curves], dtype=torch.int32)
    return TensorDataset(flux_stack, time_stack, ticids, mask_stack)


def load_training_catalog(
    catalog_path: Path,
    *,
    seq_len: int = DEFAULT_SEQ_LEN,
    label_column: str = "label",
    path_column: str = "path",
    label_map: Optional[Dict[str, int]] = None,
) -> Tuple[TensorDataset, Dict[str, int]]:
    df = pd.read_csv(catalog_path)
    if df.empty:
        raise ValueError("Training catalog is empty")

    base_dir = catalog_path.parent

    if label_map is None:
        unique_labels = sorted(df[label_column].unique())
        label_map = {label: idx for idx, label in enumerate(unique_labels)}

    fluxes: List[LoadedCurve] = []
    label_indices: List[int] = []

    for _, row in df.iterrows():
        label = row[label_column]
        if label not in label_map:
            raise ValueError(f"Unknown label '{label}' encountered. Known labels: {list(label_map)}")

        file_path = Path(row[path_column])
        if not file_path.is_absolute():
            file_path = (base_dir / file_path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"Light curve path does not exist: {file_path}")

        curve = load_processed_lightcurve(file_path, seq_len=seq_len)
        if not curve.mask.any():
            raise ValueError(f"Processed light curve is empty: {file_path}")

        fluxes.append(curve)
        label_indices.append(label_map[label])

    num_classes = len(label_map)
    one_hot = torch.nn.functional.one_hot(torch.tensor(label_indices), num_classes=num_classes).to(torch.float32)

    dataset = _build_dataset_from_curves(fluxes, labels=one_hot)
    return dataset, label_map


def load_inference_directory(
    directory: Path,
    *,
    seq_len: int = DEFAULT_SEQ_LEN,
    recursive: bool = True,
) -> TensorDataset:
    """Load light curves from a directory of .txt files."""
    if recursive:
        txt_paths = sorted(directory.rglob("*.txt"))
    else:
        txt_paths = sorted(directory.glob("*.txt"))

    if not txt_paths:
        raise FileNotFoundError(f"No .txt files found in {directory}")

    curves: List[LoadedCurve] = []
    for path in txt_paths:
        curve = load_processed_lightcurve(path, seq_len=seq_len)
        if curve.ticid == -1:
            curve.ticid = len(curves)
        curves.append(curve)

    return _build_dataset_from_curves(curves)
