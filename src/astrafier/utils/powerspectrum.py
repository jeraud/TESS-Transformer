#!/usr/bin/env python3
"""Power spectrum utilities ported from the original helper module."""

from __future__ import annotations

import os
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import lightkurve
from astropy.timeseries import LombScargle
from bottleneck import nanmedian, nanmean, nanmax, nanmin
from scipy.optimize import minimize_scalar
from scipy.integrate import simpson


class powerspectrum(object):
    """Wrapper around :class:`astropy.timeseries.LombScargle`."""

    def __init__(self, lightcurve, fit_mean: bool = False):
        self.fit_mean = fit_mean
        indx = np.isfinite(lightcurve.flux)
        self.df = 1 / (86400 * (nanmax(lightcurve.time[indx]) - nanmin(lightcurve.time[indx])))
        self.nyquist = 1 / (2 * 86400 * nanmedian(np.diff(lightcurve.time[indx])))
        self.standard = None
        self.ls = LombScargle(
            lightcurve.time[indx] * 86400,
            lightcurve.flux[indx],
            center_data=True,
            fit_mean=self.fit_mean,
        )
        self.df = self.fundamental_spacing_integral()
        self.standard = self.powerspectrum(oversampling=1, nyquist_factor=1, scale=None)
        N = len(self.ls.t)
        tot_MS = np.sum((self.ls.y - nanmean(self.ls.y)) ** 2) / N
        tot_lomb = np.sum(self.standard[1])
        self.normfactor = tot_MS / tot_lomb
        self.standard = list(self.standard)
        self.standard[1] *= self.normfactor / (self.df * 1e6)
        self.standard = tuple(self.standard)

    def copy(self):
        return deepcopy(self)

    def fundamental_spacing_minimum(self):
        freq_cen = 0.5 * self.nyquist
        x = 0.5 * np.sin(2 * np.pi * freq_cen * self.ls.t) + 0.5 * np.cos(2 * np.pi * freq_cen * self.ls.t)
        ls = LombScargle(self.ls.t, x, center_data=True, fit_mean=self.fit_mean)
        window = lambda freq: ls.power(freq_cen + freq, normalization="psd", method="fast")
        res = minimize_scalar(window, [0.75 * self.df, self.df, 1.25 * self.df])
        df = res.x
        return df

    def fundamental_spacing_integral(self):
        freq, window = self.windowfunction(width=100 * self.df, oversampling=5)
        df = simpson(window, freq)
        return df * 1e-6

    def powerspectrum(self, freq=None, oversampling: float = 1, nyquist_factor: float = 1, scale: str | None = "power"):
        assume_regular_frequency = False
        if freq is None:
            if scale == "powerdensity" and oversampling == 1 and nyquist_factor == 1 and self.standard:
                return self.standard
            freq = np.arange(self.df / oversampling, nyquist_factor * self.nyquist, self.df / oversampling, dtype="float64")
            assume_regular_frequency = True

        power = self.ls.power(freq, normalization="psd", method="fast", assume_regular_frequency=assume_regular_frequency)
        power = np.clip(power, 0, None)
        freq *= 1e6
        if scale is None:
            pass
        elif scale == "power":
            power *= self.normfactor * 2
        elif scale == "powerdensity":
            power *= self.normfactor / (self.df * 1e6)
        elif scale == "amplitude":
            power = np.sqrt(power * self.normfactor * 2)

        return freq, power

    def windowfunction(self, width: float | None = None, oversampling: float = 10):
        if width is None:
            width = 100 * self.df

        freq_cen = 0.5 * self.nyquist
        Nfreq = int(oversampling * width / self.df)
        freq = freq_cen + (self.df / oversampling) * np.arange(-Nfreq, Nfreq, 1)

        x = 0.5 * np.sin(2 * np.pi * freq_cen * self.ls.t) + 0.5 * np.cos(2 * np.pi * freq_cen * self.ls.t)

        ls = LombScargle(self.ls.t, x, center_data=True, fit_mean=self.fit_mean)
        power = ls.power(freq, method="fast", normalization="psd", assume_regular_frequency=True)
        power /= power[int(len(power) / 2)]

        freq -= freq_cen
        freq *= 1e6
        return freq, power

    def plot(self, ax=None, xlabel: str = "Frequency (muHz)", ylabel: str | None = None, style: str | None = "powerspectrum"):
        if ylabel is None:
            ylabel = {
                "powerdensity": "Power density (ppm^2/muHz)",
                "power": "Power (ppm^2)",
                "amplitude": "Amplitude (ppm)",
            }["powerdensity"]

        if style is None or style == "powerspectrum":
            style = os.path.join(os.path.dirname(__file__), "powerspectrum.mplstyle")
        with plt.style.context(style):
            if ax is None:
                fig, ax = plt.subplots(1)

            ax.loglog(self.standard[0], self.standard[1], "k-")
            ax.set_xlabel("Frequency (muHz)")
            ax.set_ylabel(ylabel)
            ax.set_xlim(self.standard[0][0], self.standard[0][-1])

    def optimize_peak(self, fmax):
        fmax = np.atleast_1d(fmax)
        if len(fmax) == 3:
            freq_low, fmax, freq_high = fmax
        else:
            fmax = fmax[0]
            freq_low = fmax - 2 * self.df * 1e6
            freq_high = fmax + 2 * self.df * 1e6

        freq_low = np.clip(freq_low, 0.25 * self.df * 1e6, None)
        func = lambda f: -self.ls.power(f * 1e-6, method="fast", normalization="psd", assume_regular_frequency=False)
        res = minimize_scalar(func, bracket=[freq_low, fmax, freq_high], bounds=(freq_low, freq_high), method="bounded", options={"xatol": 1e-5})
        return res.x

    def alpha_beta(self, freq):
        alpha, beta = self.ls.model_parameters(freq * 1e-6, units=False)
        return alpha, beta

    def model(self, a, b, freq):
        omegax = 0.1728 * np.pi * freq * self.ls.t
        return a * np.sin(omegax) + b * np.cos(omegax)

    def false_alarm_probability(self, freq):
        p_harmonic = self.ls.power(freq * 1e-6, method="fast")
        return self.ls.false_alarm_probability(p_harmonic)

    def replace_lightcurve(self, lightcurve):
        indx = np.isfinite(lightcurve.flux)
        self.ls = LombScargle(lightcurve.time[indx] * 86400, lightcurve.flux[indx], center_data=True, fit_mean=self.fit_mean)

