"""Plotting helpers for light curves and power spectra."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .powerspectrum import powerspectrum


def create_plot(lc, filename: str, title: str, save: bool = True):
    flux = lc.flux.value
    time = lc.time.value

    if flux.shape == (0,):
        flux = np.zeros(4000)
        time = np.zeros(4000)

    lc_df = pd.DataFrame({"flux": flux, "time": time})
    ps = powerspectrum(lc_df)
    frequency, power = ps.powerspectrum(scale="amplitude", oversampling=15)

    fig = plt.figure(figsize=(19, 9.5))
    ax = plt.subplot(211)

    sorted_df = lc_df.sort_values("time")
    plt.title(title)
    plt.plot(sorted_df.time, sorted_df.flux, color="dimgrey", linewidth=0.6)
    plt.xlabel("time [days]")
    plt.ylabel("flux [ppm]")

    ax = plt.subplot(212)
    plt.plot(frequency, power, color="red", linewidth=1.2)
    ax.set_xlabel(r"frequency [$\mu$Hz]")
    ax.set_xlim((0, 280))
    ax.set_ylim((0, np.max(power)))

    ax2 = plt.twiny()
    ax2.set_xticks(ax.get_xticks().tolist())
    ax.set_xlim((0, 280))
    conversion = 1 / 86400 * 1e6
    labels = [f"{tick / conversion:.2f} c/d" for tick in ax2.get_xticks()]
    ax2.set_xticklabels(labels)

    plt.tight_layout()
    if save:
        plt.savefig(
            filename,
            dpi=50,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.close()
    else:
        return fig

