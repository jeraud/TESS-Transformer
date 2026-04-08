"""Preprocessing entry points for the Astrafier CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from tqdm import tqdm

from astrafier.data.loading import load_training_catalog, DEFAULT_SEQ_LEN


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess FITS light curves into .pt tensors for faster training."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="CSV with 'label' and 'path' columns (same format as --train-csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .pt file path",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Column name for class labels (default: label)",
    )
    parser.add_argument(
        "--path-column",
        type=str,
        default="path",
        help="Column name for FITS file paths (default: path)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help=f"Sequence length after preprocessing (default: {DEFAULT_SEQ_LEN})",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    csv_path = args.csv.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading and preprocessing from {csv_path}...")
    dataset, label_map = load_training_catalog(
        csv_path,
        seq_len=args.seq_len,
        label_column=args.label_column,
        path_column=args.path_column,
    )

    # Extract tensors from dataset
    flux = dataset.tensors[0]
    time = dataset.tensors[1]
    labels = dataset.tensors[2]
    mask = dataset.tensors[3]

    torch.save({
        "flux": flux,
        "time": time,
        "labels": labels,
        "mask": mask,
        "label_map": label_map,
        "seq_len": args.seq_len,
    }, output_path)

    print(f"Saved {len(flux)} samples to {output_path}")

    # Print class distribution
    print("\nClass distribution:")
    counts = labels.sum(dim=0).int().tolist()
    for class_name, count in sorted(label_map.items(), key=lambda x: x[1]):
        print(f"  {class_name}: {counts[label_map[class_name]]}")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess FITS light curves into .pt tensors for faster training",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="CSV with 'label' and 'path' columns (same format as --train-csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .pt file path",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Column name for class labels (default: label)",
    )
    parser.add_argument(
        "--path-column",
        type=str,
        default="path",
        help="Column name for FITS file paths (default: path)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help=f"Sequence length after preprocessing (default: {DEFAULT_SEQ_LEN})",
    )
    parser.set_defaults(func=run)


if __name__ == "__main__":
    main()
