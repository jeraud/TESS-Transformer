"""Training entry points for the Astrafier CLI."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import CSVLogger

import pandas as pd

from astrafier.constants import CLASS_NAMES
from astrafier.data.loading import load_training_catalog
from astrafier.models import AstrafierModule
from astrafier.utils.training import AccuracyLogger
from huggingface_hub import hf_hub_download

DEFAULT_CLASS_NAMES = list(CLASS_NAMES)
DEFAULT_VAL_SPLIT = 0.1
DEFAULT_TEST_SPLIT = 0.2
DEFAULT_TRAIN_SPLIT = 1.0 - DEFAULT_VAL_SPLIT - DEFAULT_TEST_SPLIT


def _load_tensor_file(path: Path) -> Dict[str, object]:
    """Load tensors from .pt or .safetensors file."""
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        import json

        tensors = load_file(path)
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            tensors.update(meta)
        return tensors
    else:
        return torch.load(path, weights_only=False)


def load_preprocessed_tensors(
    train_pt: Path, test_pt: Path, val_split: float, split_seed: int
) -> Tuple[TensorDataset, TensorDataset, TensorDataset, torch.Tensor, Dict[str, int]]:
    """Load pre-processed .pt or .safetensors files and create datasets.

    Creates a validation split from the training data.
    """
    train_data = _load_tensor_file(train_pt)
    test_data = _load_tensor_file(test_pt)

    train_flux = train_data["flux"]
    train_time = train_data["time"]
    train_labels = train_data["labels"]
    train_mask = train_data["mask"]
    label_map = train_data["label_map"]

    test_flux = test_data["flux"]
    test_time = test_data["time"]
    test_labels = test_data["labels"]
    test_mask = test_data["mask"]

    # Split training into train/val
    num_train = len(train_flux)
    indices = np.arange(num_train)
    label_indices = torch.argmax(train_labels, dim=1).cpu().numpy()

    if val_split > 0:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_split,
            random_state=split_seed,
            shuffle=True,
            stratify=label_indices,
        )
    else:
        train_idx = indices
        val_idx = np.array([], dtype=np.int64)

    # Create datasets
    dataset_train = TensorDataset(
        train_flux[train_idx],
        train_time[train_idx],
        train_labels[train_idx],
        train_mask[train_idx],
    )
    dataset_val = TensorDataset(
        train_flux[val_idx],
        train_time[val_idx],
        train_labels[val_idx],
        train_mask[val_idx],
    ) if len(val_idx) > 0 else TensorDataset(
        torch.empty(0, train_flux.shape[1]),
        torch.empty(0, train_time.shape[1]),
        torch.empty(0, train_labels.shape[1]),
        torch.empty(0, train_mask.shape[1], dtype=torch.bool),
    )
    dataset_test = TensorDataset(test_flux, test_time, test_labels, test_mask)

    # Compute class weights from training set
    class_counts = train_labels[train_idx].sum(dim=0).clamp_min(1.0)
    total = max(1, len(train_idx))
    class_weights = torch.full_like(class_counts, float(total)) / class_counts

    return dataset_train, dataset_val, dataset_test, class_weights, label_map


def make_dataloader(dataset: Subset | TensorDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=shuffle)


def _validate_splits(train_split: float, val_split: float, test_split: float) -> None:
    splits = (train_split, val_split, test_split)
    if any(split < 0 for split in splits):
        raise ValueError(f"Train/val/test ratios must be non-negative, received {splits}")

    total = sum(splits)
    if not np.isclose(total, 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError(
            f"Train/val/test ratios must sum to 1.0, received {splits} (total={total:.6f})"
        )

    if train_split <= 0:
        raise ValueError("Training split must be greater than zero.")
    if val_split <= 0:
        raise ValueError("Validation split must be greater than zero.")


def _validate_mc_dropout_args(
    parser: argparse.ArgumentParser, mc_dropout: bool | None, mc_samples: int | None
) -> None:
    if mc_samples is not None and mc_samples < 1:
        parser.error("--mc-samples must be a positive integer")
    if mc_dropout is False and mc_samples not in (None, 1):
        parser.error("--mc-samples > 1 requires Monte Carlo dropout to be enabled")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Astrafier model from FITS light curves.")

    parser.add_argument("--train-csv", type=Path, help="CSV with label, path, and TIC columns (full pipeline)")
    parser.add_argument("--train-pt", type=Path, help="Pre-processed training .pt file")
    parser.add_argument("--test-pt", type=Path, help="Pre-processed test .pt file")
    parser.add_argument("--label-column", type=str, default="label", help="Column name for class labels")
    parser.add_argument("--path-column", type=str, default="path", help="Column name containing FITS paths")
    parser.add_argument("--tic-column", type=str, default="TIC", help="Column name for TIC IDs (for TIC-based splitting)")
    parser.add_argument("--save-preprocessed", type=Path, default=None, help="Save preprocessed tensors to this directory")
    parser.add_argument("--seq-len", type=int, default=1171, help="Sequence length after preprocessing")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training (default: %(default)s)")
    parser.add_argument("--max-epochs", type=int, default=250, help="Maximum number of training epochs")
    parser.add_argument("--min-epochs", type=int, default=5, help="Minimum number of training epochs")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Precision to use in Lightning trainer")
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        help="Distributed strategy (default: auto). Override for multi-node or custom setups.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to store checkpoints and artifacts (default: %(default)s)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=DEFAULT_TRAIN_SPLIT,
        help="Proportion of samples allocated to the training partition (default: %(default)s)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=DEFAULT_VAL_SPLIT,
        help="Proportion of samples allocated to the validation partition (default: %(default)s)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=DEFAULT_TEST_SPLIT,
        help="Proportion of samples allocated to the test partition (default: %(default)s)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed controlling deterministic dataset partitioning.",
    )
    parser.add_argument(
        "--load-from-hf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load model weights from HuggingFace Hub instead of training from scratch (default: True)",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default="paulg9/astrafier",
        help="HuggingFace repository ID (default: %(default)s)",
    )
    parser.add_argument(
        "--hf-filename",
        type=str,
        default="model.ckpt",
        help="Filename in HuggingFace repository (default: %(default)s)",
    )
    parser.add_argument(
        "--mc-dropout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Monte Carlo dropout for prediction loops saved in checkpoints (default: True)",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=20,
        help="Number of Monte Carlo dropout samples to use when enabled (default: %(default)s)",
    )

    args = parser.parse_args(argv)
    _validate_mc_dropout_args(parser, args.mc_dropout, args.mc_samples)

    # Validate data source arguments
    has_pt = args.train_pt is not None or args.test_pt is not None
    has_csv = args.train_csv is not None

    if has_pt and has_csv:
        parser.error("Cannot specify both --train-csv and --train-pt/--test-pt. Use one approach.")
    if has_pt:
        if args.train_pt is None or args.test_pt is None:
            parser.error("Both --train-pt and --test-pt must be provided together.")
    elif not has_csv:
        parser.error("Must provide either --train-csv or --train-pt/--test-pt.")

    return args


def build_datasets_from_csv(
    args: argparse.Namespace,
) -> Tuple[TensorDataset, TensorDataset, TensorDataset, torch.Tensor, Dict[str, int]]:
    """Build datasets from CSV with train/test splitting.

    If TIC column exists, splits by TIC to avoid data leakage.
    Otherwise, falls back to row-based splitting with a warning.
    """
    df = pd.read_csv(args.train_csv)

    tic_col = args.tic_column
    label_col = args.label_column
    use_tic_split = tic_col in df.columns

    if use_tic_split:
        # TIC-based splitting (recommended)
        tic_labels = df.groupby(tic_col)[label_col].agg(lambda x: x.mode().iloc[0]).reset_index()
        unique_tics = tic_labels[tic_col].values
        labels_for_stratify = tic_labels[label_col].values

        train_tics, test_tics = train_test_split(
            unique_tics,
            test_size=args.test_split,
            random_state=args.split_seed,
            shuffle=True,
            stratify=labels_for_stratify,
        )

        train_tic_set = set(train_tics)
        test_tic_set = set(test_tics)

        overlap = train_tic_set & test_tic_set
        if overlap:
            raise RuntimeError(f"BUG: Found {len(overlap)} overlapping TICs")

        print(f"TIC-based split: {len(train_tics)} train TICs, {len(test_tics)} test TICs (zero overlap)")

        train_rows = df[df[tic_col].isin(train_tic_set)].index.tolist()
        test_rows = df[df[tic_col].isin(test_tic_set)].index.tolist()
    else:
        # Row-based splitting (fallback)
        print(
            f"WARNING: No '{tic_col}' column found. Using row-based splitting. "
            f"If the same source appears multiple times, this may cause data leakage."
        )
        labels_for_stratify = df[label_col].values
        all_indices = np.arange(len(df))

        train_rows, test_rows = train_test_split(
            all_indices,
            test_size=args.test_split,
            random_state=args.split_seed,
            shuffle=True,
            stratify=labels_for_stratify,
        )
        train_rows = train_rows.tolist()
        test_rows = test_rows.tolist()

    print(f"Row counts: {len(train_rows)} train, {len(test_rows)} test")

    # Load and preprocess all data
    dataset, label_map = load_training_catalog(
        args.train_csv,
        seq_len=args.seq_len,
        label_column=args.label_column,
        path_column=args.path_column,
    )

    # Extract tensors for train split
    train_flux = dataset.tensors[0][train_rows]
    train_time = dataset.tensors[1][train_rows]
    train_labels_tensor = dataset.tensors[2][train_rows]
    train_mask_tensor = dataset.tensors[3][train_rows]

    # Extract tensors for test split
    test_flux = dataset.tensors[0][test_rows]
    test_time = dataset.tensors[1][test_rows]
    test_labels_tensor = dataset.tensors[2][test_rows]
    test_mask_tensor = dataset.tensors[3][test_rows]

    # Carve validation from training (by row, within train TICs - this is fine)
    num_train = len(train_rows)
    indices = np.arange(num_train)
    label_indices = torch.argmax(train_labels_tensor, dim=1).cpu().numpy()

    if args.val_split > 0:
        actual_train_idx, val_idx = train_test_split(
            indices,
            test_size=args.val_split,
            random_state=args.split_seed,
            shuffle=True,
            stratify=label_indices,
        )
    else:
        actual_train_idx = indices
        val_idx = np.array([], dtype=np.int64)

    # Create final datasets
    dataset_train = TensorDataset(
        train_flux[actual_train_idx],
        train_time[actual_train_idx],
        train_labels_tensor[actual_train_idx],
        train_mask_tensor[actual_train_idx],
    )

    if len(val_idx) > 0:
        dataset_val = TensorDataset(
            train_flux[val_idx],
            train_time[val_idx],
            train_labels_tensor[val_idx],
            train_mask_tensor[val_idx],
        )
    else:
        dataset_val = TensorDataset(
            torch.empty(0, train_flux.shape[1]),
            torch.empty(0, train_time.shape[1]),
            torch.empty(0, train_labels_tensor.shape[1]),
            torch.empty(0, train_mask_tensor.shape[1], dtype=torch.bool),
        )

    dataset_test = TensorDataset(test_flux, test_time, test_labels_tensor, test_mask_tensor)

    # Compute class weights from training set
    class_counts = train_labels_tensor[actual_train_idx].sum(dim=0).clamp_min(1.0)
    total = max(1, len(actual_train_idx))
    class_weights = torch.full_like(class_counts, float(total)) / class_counts

    # Optionally save preprocessed tensors
    if args.save_preprocessed is not None:
        save_dir = args.save_preprocessed.resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save train (includes val, before val split)
        torch.save({
            "flux": train_flux,
            "time": train_time,
            "labels": train_labels_tensor,
            "mask": train_mask_tensor,
            "label_map": label_map,
            "seq_len": args.seq_len,
        }, save_dir / "train.pt")

        torch.save({
            "flux": test_flux,
            "time": test_time,
            "labels": test_labels_tensor,
            "mask": test_mask_tensor,
            "label_map": label_map,
            "seq_len": args.seq_len,
        }, save_dir / "test.pt")

        print(f"Saved preprocessed tensors to {save_dir}/train.pt and {save_dir}/test.pt")

    return dataset_train, dataset_val, dataset_test, class_weights, label_map


def run(args: argparse.Namespace) -> None:
    # Validate data source arguments
    has_pt = getattr(args, "train_pt", None) is not None or getattr(args, "test_pt", None) is not None
    has_csv = getattr(args, "train_csv", None) is not None

    if has_pt and has_csv:
        raise ValueError("Cannot specify both --train-csv and --train-pt/--test-pt. Use one approach.")
    if has_pt:
        if getattr(args, "train_pt", None) is None or getattr(args, "test_pt", None) is None:
            raise ValueError("Both --train-pt and --test-pt must be provided together.")
    elif not has_csv:
        raise ValueError("Must provide either --train-csv or --train-pt/--test-pt.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(42, workers=True)

    if args.train_pt is not None:
        dataset_train, dataset_val, dataset_test, class_weights, label_map = load_preprocessed_tensors(
            args.train_pt, args.test_pt, args.val_split, args.split_seed
        )
    else:
        dataset_train, dataset_val, dataset_test, class_weights, label_map = build_datasets_from_csv(args)
    train_loader = make_dataloader(dataset_train, args.batch_size, shuffle=True)
    val_loader = make_dataloader(dataset_val, args.batch_size, shuffle=False)
    test_loader = make_dataloader(dataset_test, args.batch_size, shuffle=False)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        save_top_k=1,
        mode="max",
        filename="{epoch:02d}-{val_acc:.2f}",
        dirpath=str(checkpoint_dir),
    )
    accuracy_logger = AccuracyLogger()

    if args.load_from_hf:
        pl.utilities.rank_zero_info(
            f"Loading model from HuggingFace Hub: {args.hf_repo_id}/{args.hf_filename}"
        )
        ckpt_path = hf_hub_download(repo_id=args.hf_repo_id, filename=args.hf_filename)
        model = AstrafierModule.load_from_checkpoint(
            ckpt_path,
            strict=False,
            class_weight=class_weights,
            mc_dropout=args.mc_dropout,
            mc_samples=args.mc_samples,
        )
        pl.utilities.rank_zero_info(f"Loaded checkpoint from {ckpt_path}")
    else:
        model = AstrafierModule(
            class_weight=class_weights,
            mc_dropout=args.mc_dropout,
            mc_samples=args.mc_samples,
        )

    logger = CSVLogger(save_dir=str(output_dir), name="lightning")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        accelerator="auto",
        devices="auto",
        precision=args.precision,
        gradient_clip_val=1.0,
        strategy=args.strategy,
        callbacks=[checkpoint_callback, accuracy_logger],
        default_root_dir=str(output_dir),
        logger=logger,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_ckpt_path = checkpoint_callback.best_model_path or None
    eval_model = (
        AstrafierModule.load_from_checkpoint(
            best_ckpt_path,
            strict=False,
            class_weight=class_weights,
            mc_dropout=args.mc_dropout,
            mc_samples=args.mc_samples,
        )
        if best_ckpt_path
        else model
    )

    val_metrics = []
    if len(dataset_val) > 0:
        val_metrics = trainer.validate(model=eval_model, dataloaders=val_loader, verbose=False)
        if val_metrics:
            pl.utilities.rank_zero_info(f"Validation metrics: {val_metrics[0]}")

    test_metrics = []
    if len(dataset_test) > 0:
        test_metrics = trainer.test(model=eval_model, dataloaders=test_loader, verbose=False)
        if test_metrics:
            pl.utilities.rank_zero_info(f"Test metrics: {test_metrics[0]}")
    rounded_test_acc = 0.0
    if test_metrics:
        rounded_test_acc = round(float(test_metrics[0].get("test_acc_epoch", 0.0)), 4)
        pl.utilities.rank_zero_info(
            f"Test Accuracy: {test_metrics[0].get('test_acc_epoch')}, "
            f"Test Loss: {test_metrics[0].get('test_loss_epoch')}"
        )
    output_path = output_dir / f"cl_model_{rounded_test_acc}.ckpt"
    os.makedirs(output_path.parent, exist_ok=True)
    if best_ckpt_path:
        shutil.copy(best_ckpt_path, output_path)
    else:
        trainer.save_checkpoint(str(output_path))

    if len(dataset_test) > 0:
        metric_device = "cuda" if torch.cuda.is_available() else "cpu"
        metric = eval_model.conf_matrix.to(metric_device)
        class_names = [label for label, _ in sorted(label_map.items(), key=lambda item: item[1])]
        if metric.num_classes != len(class_names):
            known = [name for name in CLASS_NAMES if name not in class_names]
            class_names.extend(known)
            class_names = class_names[: metric.num_classes]
        fig_, ax_ = metric.plot(labels=class_names)
        confusion_path = output_dir / f"confusion_matrix_{rounded_test_acc}.png"
        fig_.savefig(confusion_path)
        plt.close(fig_)
        pl.utilities.rank_zero_info(f"Saved confusion matrix to {confusion_path}")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("train", help="Train the Astrafier model from FITS light curves")
    parser.add_argument("--train-csv", type=Path, help="CSV with columns for labels and FITS paths")
    parser.add_argument("--train-pt", type=Path, help="Pre-processed training .pt file (from astrafier preprocess)")
    parser.add_argument("--test-pt", type=Path, help="Pre-processed test .pt file (from astrafier preprocess)")
    parser.add_argument("--label-column", type=str, default="label", help="Column name for class labels")
    parser.add_argument("--path-column", type=str, default="path", help="Column name with FITS paths")
    parser.add_argument("--tic-column", type=str, default="TIC", help="Column name for TIC IDs (for TIC-based splitting)")
    parser.add_argument("--save-preprocessed", type=Path, default=None, help="Save preprocessed tensors to this directory")
    parser.add_argument("--seq-len", type=int, default=1171, help="Sequence length after preprocessing")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training (default: %(default)s)")
    parser.add_argument("--max-epochs", type=int, default=250, help="Maximum number of training epochs")
    parser.add_argument("--min-epochs", type=int, default=5, help="Minimum number of training epochs")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Precision to use in Lightning trainer")
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        help="Distributed strategy (default: auto). Override for specific cluster setups.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to store checkpoints and artifacts (default: %(default)s)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=DEFAULT_TRAIN_SPLIT,
        help="Proportion of samples allocated to the training partition (default: %(default)s)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=DEFAULT_VAL_SPLIT,
        help="Proportion of samples allocated to the validation partition (default: %(default)s)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=DEFAULT_TEST_SPLIT,
        help="Proportion of samples allocated to the test partition (default: %(default)s)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed controlling deterministic dataset partitioning.",
    )
    parser.add_argument(
        "--load-from-hf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load model weights from HuggingFace Hub instead of training from scratch (default: True)",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default="paulg9/astrafier",
        help="HuggingFace repository ID (default: %(default)s)",
    )
    parser.add_argument(
        "--hf-filename",
        type=str,
        default="model.ckpt",
        help="Filename in HuggingFace repository (default: %(default)s)",
    )
    parser.add_argument(
        "--mc-dropout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Monte Carlo dropout for prediction loops saved in checkpoints (default: True)",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=20,
        help="Number of Monte Carlo dropout samples to use when enabled (default: %(default)s)",
    )

    parser.set_defaults(func=run)


if __name__ == "__main__":
    main()

