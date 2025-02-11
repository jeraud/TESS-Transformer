import os
import numpy as np
from astropy.io import fits
from extract_from_fits import extract_qlp
from preprocess import preprocess_lightcurve
from load_qlp import load_qlp_tensors
import lightkurve as lk 
import torch
SECTOR_26 = '/Users/paul/Desktop/UROP/data/sector_26_curves/'
# LEN_SECTOR_26 = len(os.listdir(SECTOR_26))

flux, time, ticids = load_qlp_tensors(SECTOR_26, 1171, False, None, True)

torch.save(flux, 'sector26_flux_tensor.pt')
torch.save(time, 'sector26_time_tensor.pt')
torch.save(ticids, 'sector26_ticids_tensor.pt')