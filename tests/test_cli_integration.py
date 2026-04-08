from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from astropy.io import fits

from astrafier.commands import predict as predict_cmd
from astrafier.commands import train as train_cmd


def _make_fits(path: Path, ticid: int, length: int = 50) -> None:
    time = np.linspace(0, 10, length).astype(np.float64)
    flux = (np.sin(time) + 1).astype(np.float64)
    quality = np.zeros(length, dtype=np.int16)

    primary = fits.PrimaryHDU()
    primary.header["TICID"] = ticid
    cols = [
        fits.Column(name="TIME", format="D", array=time),
        fits.Column(name="SAP_FLUX", format="D", array=flux),
        fits.Column(name="QUALITY", format="J", array=quality),
    ]
    table = fits.BinTableHDU.from_columns(cols)
    fits.HDUList([primary, table]).writeto(path, overwrite=True)


def test_train_cli_smoke(tmp_path, monkeypatch):
    fits_dir = tmp_path / "fits"
    fits_dir.mkdir()
    file1 = fits_dir / "curve1.fits"
    file2 = fits_dir / "curve2.fits"
    _make_fits(file1, ticid=1234)
    _make_fits(file2, ticid=5678)

    paths = [file1.relative_to(tmp_path)] * 10 + [file2.relative_to(tmp_path)] * 10
    labels = ["APERIODIC"] * 10 + ["CONTACT_ROT"] * 10

    train_csv = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "label": labels,
            "path": paths,
        }
    ).to_csv(train_csv, index=False)

    args = SimpleNamespace(
        train_csv=train_csv,
        train_pt=None,
        test_pt=None,
        label_column="label",
        path_column="path",
        tic_column="TIC",
        save_preprocessed=None,
        seq_len=64,
        batch_size=4,
        max_epochs=1,
        min_epochs=1,
        precision="32-true",
        strategy="auto",
        output_dir=tmp_path / "artifacts",
        train_split=0.7,
        val_split=0.1,
        test_split=0.2,
        split_seed=42,
        load_from_hf=False,
        hf_repo_id="stub/repo",
        hf_filename="model.ckpt",
        mc_dropout=True,
        mc_samples=20,
    )

    best_ckpt = tmp_path / "best.ckpt"
    best_ckpt.write_text("stub")
    dummy_eval_model = train_cmd.AstrafierModule(class_weight=None)

    monkeypatch.setattr(
        train_cmd.AstrafierModule,
        "load_from_checkpoint",
        classmethod(lambda cls, *a, **k: dummy_eval_model),
    )

    class DummyCheckpoint:
        def __init__(self, *args, **kwargs):
            self.best_model_path = str(best_ckpt)

    monkeypatch.setattr(train_cmd.pl.callbacks, "ModelCheckpoint", DummyCheckpoint)

    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            pass
        
        def validate(self, model, dataloaders, verbose=False):
            return [{"val_acc": 0.75}]

        def test(self, model, dataloaders, verbose=False):
            preds = torch.zeros(1, dtype=torch.long)
            target = torch.zeros(1, dtype=torch.long)
            model.conf_matrix.update(preds, target)
            return [{"test_acc_epoch": 0.5, "test_loss_epoch": 1.0}]

        def save_checkpoint(self, path):
            Path(path).touch()

    monkeypatch.setattr(train_cmd.pl, "Trainer", DummyTrainer)

    train_cmd.run(args)

    assert (args.output_dir / "confusion_matrix_0.5.png").exists()


def test_predict_cli_smoke(tmp_path, monkeypatch):
    fits_dir = tmp_path / "fits"
    fits_dir.mkdir()
    _make_fits(fits_dir / "curve.fits", ticid=3210)

    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("stub")

    args = SimpleNamespace(
        checkpoint=checkpoint,
        fits_dir=fits_dir,
        seq_len=64,
        recursive=False,
        batch_size=2,
        output_dir=tmp_path / "predictions",
        load_from_hf=False,
        hf_repo_id="stub/repo",
        hf_filename="model.ckpt",
        mc_dropout=None,
        mc_samples=None,
        precision="32-true",
    )

    class DummyPredictModel:
        def predict_step(self, batch, batch_idx=None):
            flux, time, ticids, mask = batch
            outputs = []
            for tic in ticids:
                outputs.append({"tic": int(tic.item()), "probabilities": [1.0] * 8})
            return outputs

    monkeypatch.setattr(
        predict_cmd.AstrafierModule,
        "load_from_checkpoint",
        classmethod(lambda cls, *a, **k: DummyPredictModel()),
    )

    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, model, dataloaders):
            outputs = []
            for idx, batch in enumerate(dataloaders):
                outputs.append(model.predict_step(batch, idx))
            return outputs

    monkeypatch.setattr(predict_cmd.pl, "Trainer", DummyTrainer)

    predict_cmd.run(args)

    output_file = args.output_dir / "predictions.csv"
    output_file = args.output_dir / "predictions.csv"
    assert output_file.exists()
    assert output_file.read_text().strip() != ""


def test_validate_splits_requires_validation():
    with pytest.raises(ValueError):
        train_cmd._validate_splits(0.8, 0.0, 0.2)


def test_predict_cli_requires_dropout_for_mc_samples(tmp_path):
    fits_dir = tmp_path / "fits"
    fits_dir.mkdir()
    dummy_ckpt = tmp_path / "model.ckpt"
    dummy_ckpt.write_text("stub")

    with pytest.raises(SystemExit):
        predict_cmd.parse_args(
            [
                "--fits-dir",
                str(fits_dir),
                "--checkpoint",
                str(dummy_ckpt),
                "--mc-samples",
                "5",
                "--no-mc-dropout",
            ]
        )

