"""Inference entry points for the Astrafier CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List
import csv
import json

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from astrafier.constants import CLASS_NAMES
from astrafier.data.loading import load_inference_directory
from astrafier.models import AstrafierModule
from huggingface_hub import hf_hub_download


def _default_precision() -> str:
    try:
        if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return "bf16-mixed"
    except Exception:
        pass
    return "32-true"


def _validate_mc_dropout_args(
    parser: argparse.ArgumentParser, mc_dropout: bool | None, mc_samples: int | None
) -> None:
    if mc_samples is not None and mc_samples < 1:
        parser.error("--mc-samples must be a positive integer")
    if mc_dropout is False and mc_samples not in (None, 1):
        parser.error("--mc-samples > 1 requires Monte Carlo dropout to be enabled")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with trained Astrafier weights.")
    parser.add_argument("--checkpoint", type=Path, help="Path to a trained checkpoint (.ckpt)")
    parser.add_argument("--fits-dir", type=Path, required=True, help="Directory containing FITS light curves")
    parser.add_argument("--seq-len", type=int, default=1171, help="Sequence length after preprocessing")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for FITS files in the directory")
    parser.add_argument("--batch-size", type=int, default=128, help="Prediction batch size (default: %(default)s)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to store predictions (default: %(default)s)",
    )
    parser.add_argument(
        "--load-from-hf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load model weights from HuggingFace Hub instead of local checkpoint (default: True)",
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
        default=False,
        help="Enable Monte Carlo dropout during inference (default: disabled)",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=None,
        help="Override the number of Monte Carlo samples (default: checkpoint value)",
    )

    parser.add_argument(
        "--precision",
        type=str,
        default=_default_precision(),
        help="Precision to use for inference (default auto-detects bf16 support)",
    )
    
    args = parser.parse_args(argv)
    _validate_mc_dropout_args(parser, args.mc_dropout, args.mc_samples)
    
    # Validate that either checkpoint or load-from-hf is provided
    if not args.load_from_hf and args.checkpoint is None:
        parser.error("Either --checkpoint or --load-from-hf must be provided")
    
    return args


def _resolve_checkpoint_path(args: argparse.Namespace) -> str:
    if args.load_from_hf:
        pl.utilities.rank_zero_info(f"Loading model from HuggingFace Hub: {args.hf_repo_id}/{args.hf_filename}")
        ckpt_path = hf_hub_download(repo_id=args.hf_repo_id, filename=args.hf_filename)
        pl.utilities.rank_zero_info(f"Downloaded checkpoint to {ckpt_path}")
        return ckpt_path

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint '{checkpoint}' does not exist")
    return str(checkpoint)


def _load_model(args: argparse.Namespace) -> AstrafierModule:
    ckpt_path = _resolve_checkpoint_path(args)
    load_kwargs: Dict[str, object] = {}
    if args.mc_dropout is not None:
        load_kwargs["mc_dropout"] = args.mc_dropout
    if args.mc_samples is not None:
        load_kwargs["mc_samples"] = args.mc_samples
    model = AstrafierModule.load_from_checkpoint(ckpt_path, strict=False, **load_kwargs)
    return model


def _gather_predictions(prediction_batches: List[List[Dict[str, object]]]) -> List[Dict[str, object]]:
    merged = [item for batch in prediction_batches for item in batch]
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        gathered: List[List[Dict[str, object]] | None] = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered, merged)
        merged = []
        for partial in gathered:
            if partial:
                merged.extend(partial)
    return merged


def _write_predictions(predictions: List[Dict[str, object]], output_file: Path) -> None:
    file_exists = output_file.exists()
    with output_file.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["tic", "probabilities"])
        for prediction in predictions:
            tic = prediction["tic"]
            probs = prediction["probabilities"]
            probabilities = {CLASS_NAMES[idx]: round(prob, 4) for idx, prob in enumerate(probs)}
            writer.writerow([tic, json.dumps(probabilities)])


def run(args: argparse.Namespace) -> None:
    fits_dir = args.fits_dir.resolve()
    if not fits_dir.exists():
        raise FileNotFoundError(f"FITS directory '{fits_dir}' does not exist")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model = _load_model(args)
    dataset = load_inference_directory(fits_dir, seq_len=args.seq_len, recursive=args.recursive)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, shuffle=False)

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        precision=args.precision,
        enable_checkpointing=False,
        logger=False,
    )
    prediction_batches = trainer.predict(model=model, dataloaders=data_loader)
    predictions = _gather_predictions(prediction_batches)

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        output_file = output_dir / "predictions.csv"
        _write_predictions(predictions, output_file)
        pl.utilities.rank_zero_info(f"Wrote predictions to {output_file}")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("predict", help="Run inference with Astrafier")
    parser.add_argument("--checkpoint", type=Path, help="Path to a trained checkpoint (.ckpt)")
    parser.add_argument("--fits-dir", type=Path, required=True, help="Directory containing FITS light curves")
    parser.add_argument("--seq-len", type=int, default=1171, help="Sequence length after preprocessing")
    parser.add_argument("--recursive", action="store_true", help="Recursively search the directory for FITS files")
    parser.add_argument("--batch-size", type=int, default=128, help="Prediction batch size (default: %(default)s)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to store predictions (default: %(default)s)",
    )
    parser.add_argument(
        "--load-from-hf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load model weights from HuggingFace Hub instead of local checkpoint (default: True)",
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
        default=False,
        help="Enable Monte Carlo dropout during inference (default: disabled)",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=None,
        help="Override the number of Monte Carlo samples (default: checkpoint value)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=_default_precision(),
        help="Precision to use for inference (default auto-detects bf16 support)",
    )
    parser.set_defaults(func=run)


if __name__ == "__main__":
    main()

