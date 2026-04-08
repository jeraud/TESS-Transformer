import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from pathlib import Path

import matplotlib.pyplot as plt
from types import SimpleNamespace

from astrafier import cli as astr_cli
from astrafier.data.loading import load_inference_directory, load_training_catalog
from astrafier.utils import plots as plot_utils


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


def test_load_training_catalog(tmp_path):
    fits_dir = tmp_path / "fits"
    fits_dir.mkdir()
    file1 = fits_dir / "curve1.fits"
    file2 = fits_dir / "curve2.fits"
    _make_fits(file1, ticid=1001)
    _make_fits(file2, ticid=1002)

    catalog = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "label": ["APERIODIC", "CONTACT_ROT"],
            "path": [file1.relative_to(tmp_path), file2.relative_to(tmp_path)],
        }
    ).to_csv(catalog, index=False)

    dataset, label_map = load_training_catalog(catalog)

    assert len(dataset) == 2
    flux, time, labels, mask = dataset[0]
    assert flux.shape == (1171,)
    assert time.shape == (1171,)
    assert mask.dtype == torch.bool and mask.any()
    assert set(label_map.keys()) == {"APERIODIC", "CONTACT_ROT"}


def test_load_inference_directory(tmp_path):
    fits_dir = tmp_path / "fits"
    fits_dir.mkdir()
    for idx in range(2):
        _make_fits(fits_dir / f"curve_{idx}.fits", ticid=2000 + idx, length=40 + idx)

    dataset = load_inference_directory(fits_dir)

    assert len(dataset) == 2
    flux, time, tic, mask = dataset[0]
    assert flux.shape == (1171,)
    assert tic.dtype == torch.int64
    assert mask.dtype == torch.bool


def test_cli_parser_builds():
    parser = astr_cli.build_parser()
    assert parser.prog == "astrafier"


def test_create_plot_sorts_time(monkeypatch, tmp_path):
    class DummyPowerSpectrum:
        def __init__(self, df):
            self.df = df

        def powerspectrum(self, scale="amplitude", oversampling=15):
            return np.array([0.0, 1.0]), np.array([1.0, 2.0])

    monkeypatch.setattr(plot_utils, "powerspectrum", lambda df: DummyPowerSpectrum(df))

    class DummyLightCurve:
        def __init__(self):
            self.flux = SimpleNamespace(value=np.array([3.0, 1.0, 2.0], dtype=float))
            self.time = SimpleNamespace(value=np.array([0.2, 0.1, 0.3], dtype=float))

    lc = DummyLightCurve()
    fig = plot_utils.create_plot(lc, str(tmp_path / "plot.png"), "title", save=False)
    assert np.all(np.diff(fig.axes[0].lines[0].get_xdata()) >= 0)
    plt.close(fig)

