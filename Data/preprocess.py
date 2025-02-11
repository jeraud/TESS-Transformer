import numpy as np
from scipy.ndimage import gaussian_filter1d
import lightkurve as lk

def preprocess_lightcurve(lc, seq_len=None):
    """
    preprocesses a light curve for the model
    time: time values
    flux: flux values
    max_len: length of the light curve (in time steps)
    """
    if seq_len is None:
        seq_len = lc.time.size
    
    lc = lc.head(seq_len).remove_nans().remove_outliers(sigma=10)
    t = np.array(lc.time.value) 
    t = t + -1*t[0] + 0.0001
    f = np.array(lc.flux.value) - gaussian_filter1d(np.array(lc.flux.value), 61)
    f = (f - np.median(f)) / np.std(f)
    return t, f
