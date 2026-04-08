"""Split a dataset by TIC to avoid data leakage between train and test sets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a CSV by TIC ID to ensure no overlap between train and test sets."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Input CSV with TIC and label columns",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Output directory for train.csv and test.csv (default: current directory)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of TICs to allocate to test set (default: 0.2)",
    )
    parser.add_argument(
        "--tic-column",
        type=str,
        default="TIC",
        help="Column name for TIC IDs (default: TIC)",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Column name for labels, used for stratification (default: label)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    csv_path = args.csv.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    tic_col = args.tic_column
    label_col = args.label_column

    if tic_col not in df.columns:
        raise ValueError(f"TIC column '{tic_col}' not found. Available: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. Available: {list(df.columns)}")

    # Get unique TICs with their primary label (most common label for that TIC)
    tic_labels = df.groupby(tic_col)[label_col].agg(lambda x: x.mode().iloc[0]).reset_index()
    tic_labels.columns = [tic_col, "primary_label"]

    unique_tics = tic_labels[tic_col].values
    labels_for_stratify = tic_labels["primary_label"].values

    print(f"Total rows: {len(df)}")
    print(f"Unique TICs: {len(unique_tics)}")
    print(f"Test split: {args.test_split}")

    # Split TICs (not rows) with stratification
    train_tics, test_tics = train_test_split(
        unique_tics,
        test_size=args.test_split,
        random_state=args.seed,
        shuffle=True,
        stratify=labels_for_stratify,
    )

    # Verify no overlap
    train_set = set(train_tics)
    test_set = set(test_tics)
    overlap = train_set & test_set
    if overlap:
        raise RuntimeError(f"BUG: Found {len(overlap)} overlapping TICs: {list(overlap)[:5]}...")

    print(f"Train TICs: {len(train_tics)}, Test TICs: {len(test_tics)}")
    print(f"Overlap: {len(overlap)} (verified zero)")

    # Split dataframe by TIC membership
    train_df = df[df[tic_col].isin(train_set)]
    test_df = df[df[tic_col].isin(test_set)]

    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    # Print class distribution
    print("\nClass distribution:")
    print("  Train:")
    for label, count in train_df[label_col].value_counts().items():
        print(f"    {label}: {count}")
    print("  Test:")
    for label, count in test_df[label_col].value_counts().items():
        print(f"    {label}: {count}")

    # Save
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSaved {train_path}")
    print(f"Saved {test_path}")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "split",
        help="Split a CSV by TIC ID to ensure no overlap between train and test sets",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Input CSV with TIC and label columns",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Output directory for train.csv and test.csv (default: current directory)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of TICs to allocate to test set (default: 0.2)",
    )
    parser.add_argument(
        "--tic-column",
        type=str,
        default="TIC",
        help="Column name for TIC IDs (default: TIC)",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Column name for labels, used for stratification (default: label)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.set_defaults(func=run)


if __name__ == "__main__":
    main()
