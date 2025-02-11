import numpy as np
import lightkurve as lk
from Data.extract_from_fits import extract_qlp
from Data.preprocess import preprocess_lightcurve
import os
import torch
import pandas as pd


def load_qlp_tensors(directory, seq_len, have_labels = False, targets = None, need_preprocess = True):
    """
    
    """
    if have_labels:
        assert targets is not None, "If you have labels, you must provide them"
        if targets.endswith(('.csv', '.ecsv')):
            df = pd.read_csv(targets)
            time = np.zeros((len(df), seq_len))
            flux = np.zeros((len(df), seq_len))
            labels = []
            idx = 0
            for row in df.iterrows():
                label = row[1].iloc[1]
                curve = directory + row[1].iloc[3]
                t, f, ticid = load_qlp_curve_from_file(curve, seq_len=seq_len, need_preprocess=need_preprocess)
                time[idx, :t.shape[0]] = t
                flux[idx, :f.shape[0]] = f
                labels.append(label)
                idx += 1
            time = torch.tensor(time)
            flux = torch.tensor(flux)
            labels = pd.get_dummies(labels, dtype=float)
            labels = labels.to_numpy()
            labels = torch.tensor(labels)
            return (time, flux, labels)
    else:
        time = np.zeros((len(os.listdir(directory)), seq_len))
        flux = np.zeros((len(os.listdir(directory)), seq_len))
        ticids = []
        idx = 0
        for light_curve in os.listdir(directory):
            light_curve_path = os.path.join(directory, light_curve)
            try:
                t, f, ticid = load_qlp_curve_from_file(light_curve_path, seq_len=seq_len)
                time[idx, :t.shape[0]] = t
                flux[idx, :f.shape[0]] = f
                ticids.append(ticid)
                idx += 1
            except:
                time = time[:-1]
                flux = flux[:-1]
                continue
            
        time = torch.tensor(time)
        flux = torch.tensor(flux)
        ticids = torch.tensor(ticids)
        return (time, flux, ticids)



def load_qlp_curve_from_file(path, seq_len = None, need_preprocess = True):
    """
    Loads and preprocesses a light curve from a file
    """
    if path.endswith(('.txt', '.noisy', '.sysnoise', '.clean')):
        data = np.loadtxt(path)
        if data.shape[1] == 4:
            quality = np.asarray(data[:,3], dtype='int32')
        else:
            quality = np.zeros(data.shape[0], dtype='int32')


        lightcurve = lk.LightCurve(
            time=data[:,0],
            flux=data[:,1],
            flux_err=data[:,2],
        )
    
    elif path.endswith(('.fits.gz', '.fits')) and 'qlp' in path:
        extracted_lc = extract_qlp(path)
        ticid = extracted_lc[2]
        lightcurve = lk.LightCurve(
            time=extracted_lc[0],
            flux=extracted_lc[1],
        )
        if need_preprocess:
            time, flux = preprocess_lightcurve(lc=lightcurve, seq_len=seq_len)
        return time, flux, ticid
    