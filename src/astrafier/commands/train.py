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

from astrafier.constants import CLASS_NAMES
from astrafier.data.loading import load_training_catalog
from astrafier.models import AstrafierModule
from astrafier.utils.training import AccuracyLogger
from huggingface_hub import hf_hub_download

DEFAULT_CLASS_NAMES = list(CLASS_NAMES)
DEFAULT_TRAIN_SPLIT = 0.7
DEFAULT_VAL_SPLIT = 0.1
DEFAULT_TEST_SPLIT = 0.2


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

    parser.add_argument("--train-csv", type=Path, required=True, help="CSV with columns for labels and FITS paths")
    parser.add_argument("--label-column", type=str, default="label", help="Column name for class labels")
    parser.add_argument("--path-column", type=str, default="path", help="Column name containing FITS paths")
    parser.add_argument("--seq-len", type=int, default=1171, help="Sequence length after preprocessing")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training (default: %(default)s)")
    parser.add_argument("--max-epochs", type=int, default=150, help="Maximum number of training epochs")
    parser.add_argument("--min-epochs", type=int, default=5, help="Minimum number of training epochs")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Precision to use in Lightning trainer")
    parser.add_argument("--cadence", type=str, default="30min", help="Cadence to use for training (default: %(default)s)")
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
        default=10,
        help="Number of Monte Carlo dropout samples to use when enabled (default: %(default)s)",
    )

    args = parser.parse_args(argv)
    _validate_mc_dropout_args(parser, args.mc_dropout, args.mc_samples)
    return args


def build_datasets(
    args: argparse.Namespace,
) -> Tuple[Subset, Subset, Subset, torch.Tensor, Dict[str, int]]:
    dataset, label_map = load_training_catalog(
        args.train_csv,
        seq_len=args.seq_len,
        label_column=args.label_column,
        path_column=args.path_column,
    )

    _validate_splits(args.train_split, args.val_split, args.test_split)

    num_samples = len(dataset)
    if num_samples < 3:
        raise ValueError("Training catalog must contain at least three samples to perform splitting.")

    label_indices = torch.argmax(dataset.tensors[2], dim=1).cpu().numpy()
    all_indices = np.arange(num_samples)

    if args.test_split > 0:
        train_val_idx, test_idx = train_test_split(
            all_indices,
            test_size=args.test_split,
            random_state=args.split_seed,
            shuffle=True,
            stratify=label_indices,
        )
    else:
        train_val_idx, test_idx = all_indices, np.array([], dtype=np.int64)

    if args.val_split > 0:
        val_fraction = args.val_split / (args.train_split + args.val_split)
        stratify_val = label_indices[train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_fraction,
            random_state=args.split_seed,
            shuffle=True,
            stratify=stratify_val,
        )
    else:
        train_idx, val_idx = train_val_idx, np.array([], dtype=np.int64)

    train_indices_tensor = torch.as_tensor(train_idx, dtype=torch.long)
    train_labels = dataset.tensors[2][train_indices_tensor]
    class_counts = train_labels.sum(dim=0).clamp_min(1.0)
    total = max(1, train_labels.shape[0])
    class_weights = torch.full_like(class_counts, float(total)) / class_counts

    dataset_train = Subset(dataset, train_idx.tolist())
    dataset_val = Subset(dataset, val_idx.tolist())
    dataset_test = Subset(dataset, test_idx.tolist())

    return dataset_train, dataset_val, dataset_test, class_weights, label_map


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(42, workers=True)

    dataset_train, dataset_val, dataset_test, class_weights, label_map = build_datasets(args)
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
    parser.add_argument("--train-csv", type=Path, required=True, help="CSV with columns for labels and FITS paths")
    parser.add_argument("--label-column", type=str, default="label", help="Column name for class labels")
    parser.add_argument("--path-column", type=str, default="path", help="Column name with FITS paths")
    parser.add_argument("--seq-len", type=int, default=1171, help="Sequence length after preprocessing")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training (default: %(default)s)")
    parser.add_argument("--max-epochs", type=int, default=150, help="Maximum number of training epochs")
    parser.add_argument("--min-epochs", type=int, default=5, help="Minimum number of training epochs")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Precision to use in Lightning trainer")
    parser.add_argument("--cadence", type=str, default="30min", help="Cadence to use for training (default: %(default)s)")
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
        default=10,
        help="Number of Monte Carlo dropout samples to use when enabled (default: %(default)s)",
    )

    parser.set_defaults(func=run)


if __name__ == "__main__":
    main()

