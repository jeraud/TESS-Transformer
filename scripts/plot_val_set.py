"""Plot all light curves from validation tensors to PNGs.

Usage:
  python scripts/plot_val_set.py \
    --data-dir legacy/artefacts/dev_code \
    --output-dir plots_val

This script expects the following files in --data-dir:
  - new_val_time.pt   [N, T]
  - new_val_flux.pt   [N, T]
  - new_val_mask.pt   [N, T] (bool or convertible to bool)
  - new_val_labels.pt [N, C] (optional)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

try:
    from astrafier.constants import CLASS_NAMES
except ModuleNotFoundError:  # pragma: no cover - fallback when running without package installed
    from src.astrafier.constants import CLASS_NAMES  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot validation set curves from tensors")
    p.add_argument("--data-dir", type=Path, default=Path("legacy/artefacts/dev_code"), help="Directory with new_val_*.pt tensors")
    p.add_argument("--output-dir", type=Path, default=Path("plots_val"), help="Directory to write PNGs to")
    p.add_argument("--limit", type=int, default=None, help="Optional limit on number of curves to plot")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t_path = args.data_dir / "new_val_time.pt"
    f_path = args.data_dir / "new_val_flux.pt"
    m_path = args.data_dir / "new_val_mask.pt"
    l_path = args.data_dir / "new_val_labels.pt"

    time = torch.load(t_path).to(torch.float32)
    flux = torch.load(f_path).to(torch.float32)
    mask = torch.load(m_path)
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)

    labels = None
    if l_path.exists():
        labels = torch.load(l_path).to(torch.float32)

    N = flux.shape[0]
    if args.limit is not None:
        N = min(N, args.limit)

    for i in range(N):
        t = time[i]
        f = flux[i]
        m = mask[i]
        t_valid = t[m]
        f_valid = f[m]

        title = f"val_idx={i}"
        if labels is not None and labels.dim() == 2 and labels.shape[1] == len(CLASS_NAMES):
            c = labels[i].argmax().item()
            title += f" | label={CLASS_NAMES[c]}"

        plt.figure(figsize=(9, 4))
        plt.plot(t_valid.cpu(), f_valid.cpu(), lw=0.8, color="tab:blue")
        plt.xlabel("time [d]")
        plt.ylabel("flux [norm]")
        plt.title(title)
        out = args.output_dir / f"val_{i:05d}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        plt.close()


if __name__ == "__main__":
    main()


