import numpy as np
import torch
import lightkurve as lk
import os
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
import pytorch_lightning as pl
from helper_functions.create_plots import create_plot
from sklearn.model_selection import train_test_split
import torch
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import helper_functions.utils as utils

MAX_LEN = 1171

def open_light_curve_csv(file_path):
    """ Helper function to open a light curve csv file.
    """
    # file_path = os.path.join(dir_light_curves, filename)
    df = pd.read_csv(file_path, header=None, sep='\s+', index_col=None, names=['time','flux','flux_error'])
    df = df.dropna(subset=['flux'])
    return df

def read_light_curve_file(filepath):
    if filepath.endswith('.fits'):
            #extract light curve
            extracted = extract_qlp(filepath)
            #process curve
            lc = lk.LightCurve({'time':extracted[0], 'flux':extracted[1]})
    elif filepath.endswith('.csv') or filepath.endswith('.ecsv'):
        df = open_light_curve_csv(filepath)
        lc = lk.LightCurve({'time':df['time'], 'flux':df['flux']})

    lc = lc.remove_nans().remove_outliers(sigma=10)
    t = np.array(lc.time.value) 
    t = t + -1*t[0]
    f = np.array(lc.flux.value) - gaussian_filter1d(np.array(lc.flux.value), 61)
    # print(idx)
    f = (f - np.median(f)) / np.std(f)
    return (t, f)


def extract_TGLC(curve_path):
    """
    Extract the time, fluxes from a fits file
    """
    with fits.open(curve_path, mode="readonly") as hdulist:
        #Flags to keep:
        dont_exclude = [0,64,256,1024,2048,8192]

        # read time, sap flux, quality flag
        tess_bjds = hdulist[1].data['time']
        sap_fluxes = hdulist[1].data['aperture_flux']
        qual_flags = hdulist[1].data['TESS_flags']

        # remove flagged data
        where_no_flag = np.where(np.isin(qual_flags,dont_exclude))
        tess_bjds = tess_bjds[where_no_flag]
        sap_fluxes = sap_fluxes[where_no_flag]
        return (tess_bjds, sap_fluxes)

def extract_qlp(curve_path):
    """
    Extract the time, fluxes from a fits file
    """
    with fits.open(curve_path, mode="readonly") as hdulist:
        #Flags to keep:
        dont_exclude = [0,64,256,1024,2048,8192]

        # read time, sap flux, quality flag
        tess_bjds = hdulist[1].data['TIME']
        sap_fluxes = hdulist[1].data['SAP_FLUX']
        qual_flags = hdulist[1].data['QUALITY']

        # remove flagged data
        where_no_flag = np.where(np.isin(qual_flags,dont_exclude))
        tess_bjds = tess_bjds[where_no_flag]
        sap_fluxes = sap_fluxes[where_no_flag]
        return (tess_bjds, sap_fluxes)
    
targets = '/Users/paul/Desktop/UROP/tessv1/targets_qlp.csv'
tess_data_dir = '/Users/paul/Desktop/UROP/tessv1/'
r_ceph = '/Users/paul/Desktop/UROP/tessv2_yeschen/RRLYR_CEPHEID'

def load_tess_qlp_data():
    """
    load labeled tess qlp data into tensors 
    """
    labels = []
    flux = np.zeros((5933, MAX_LEN))
    time = np.zeros((5933, MAX_LEN))
    df = pd.read_csv(targets)
    idx = 0
    for row in df.iterrows():
        label = row[1].iloc[1]
        # num_cr = 0
        # if label == 'CONTACT_ROT':
        #     num_cr += 1
        #     print('CONTACT_ROT', num_cr)
        #     time = time[:-1, :]
        #     flux = flux[:-1, :]
        #     continue
        curve = tess_data_dir + row[1].iloc[3]
        extracted = extract_qlp(curve)
        lc = lk.LightCurve({'time':extracted[0], 'flux':extracted[1]})
        # lc = lc.remove_nans().remove_outliers(sigma=10).head(MAX_LEN)
        t = np.array(lc.time.value) 
        t = t 
        # -1*t[0] + 0.0001
        f = np.array(lc.flux.value) 
        # - gaussian_filter1d(np.array(lc.flux.value), 61)
        # print(f.shape)
        # f = f.reshape(-1, 2).mean(axis=1)
        # t = t.reshape(-1, 2).mean(axis=1)
        labels.append(label)
        time[idx, :t.shape[0]] = t 
        flux[idx, :f.shape[0]] = f
        # (f - np.median(f)) / np.std(f)
        idx += 1
    
    for filename in os.listdir(r_ceph):
        if filename.startswith("hlsp_qlp") and filename.endswith('.fits'):
                    extracted = extract_qlp(os.path.join(r_ceph,filename))
                    lc = lk.LightCurve({'time':extracted[0], 'flux':extracted[1]})
                    # lc = lc.remove_nans().remove_outliers(sigma=10).head(MAX_LEN)
                    t = np.array(lc.time.value) 
                    t = t 
                    # + -1*t[0] + 0.0001
                    f = np.array(lc.flux.value) 
                    # - gaussian_filter1d(np.array(lc.flux.value), 61)
                    print(idx)
                    # f = f.reshape(-1, 2).mean(axis=1)
                    # t = t.reshape(-1, 2).mean(axis=1)
                    labels.append('RRLYR_CEPHEID')
                    time[idx, :t.shape[0]] = t 
                    flux[idx, :f.shape[0]] = f
                    # (f - np.median(f)) / np.std(f)
                    idx += 1
        
    constants = '/Users/paul/Desktop/UROP/tessv1/CONSTANT/'
    filename_list = os.listdir(constants)   
    for i, filename in enumerate(filename_list):
        try:
            df = utils.open_light_curve_csv(filename, constants)
            labels.append('CONSTANT')
            lc = lk.LightCurve({'time':df['time'], 'flux':df['flux']})
            # lc = lc.remove_nans().remove_outliers(sigma=10).head(MAX_LEN)
            # .flatten(window_length=101)
            t = np.array(lc.time.value) 
            t = t 
            # + -1*t[0] + 0.0001
            f = np.array(lc.flux.value) 
            # - gaussian_filter1d(np.array(lc.flux.value), 61)
            # - np.mean(np.array(lc.flux.value))
            # f = f.reshape(-1, 2).mean(axis=1)
            # t = t.reshape(-1, 2).mean(axis=1)
            print(idx)
            # labels.append(row[1].iloc[1])
            time[idx, :t.shape[0]] = t 
            flux[idx, :f.shape[0]] = f
            # (f - np.median(f)) / np.std(f)
            idx += 1
        except:
            print(filename)
    flux = torch.Tensor(flux)
    time = torch.Tensor(time)
    labels = pd.get_dummies(labels, dtype=float)
    labels = labels.to_numpy()
    labels = torch.tensor(labels)
    print(labels[0])
    torch.save(time, 'qlptimetensorRAW.pt')
    torch.save(flux, 'qlpfluxtensorRAW.pt')
    torch.save(labels, 'qlplabelstensorRAW.pt')
    print(flux.shape, time.shape, labels.shape)
    return (time, flux, labels)
load_tess_qlp_data()
def load_data(dir, have_labels = False):
    """
    Load data into flux, time tensors
    
    Args:
        dir(string): the path to a directory of fits files, csv files, or directories

        have_labels(bool): whether or not this data is labeled. If it is, dir should have a directory for 
                            each labeled class, with the name of the directory the label. These directories 
                            should consist of the curves, fits or csv files
    """
    if have_labels:
        labels = []

    #initialize flux, time of shape [# of light curves, time steps]
    if not have_labels:
        file_count = len([f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))])
    else:
        file_count = 0
        for d in os.listdir(dir):
            path = os.path.join(dir,d)
            file_count += len([f for f in os.listdir(path) if os.path.isfile(path)])
    
    flux = np.zeros((file_count, MAX_LEN))
    time = np.zeros((file_count, MAX_LEN))

    idx = 0

    if have_labels:
        for label in os.listdir(dir):
            item_path = os.path.join(dir, label)
            if os.path.isdir(item_path):
                for filename in os.listdir(item_path):
                    data = read_light_curve_file(filepath=filename)
                    time[idx, :data[0].shape[0]] = data[0] 
                    flux[idx, :data[1].shape[0]] = data[1]
                    labels.append(label)
                    idx += 1
    
    # no labels
    else:
        for filename in os.listdir(dir):
            lc_path = os.path.join(dir, filename)
            data = read_light_curve_file(filepath=filename)
            time[idx, :data[0].shape[0]] = data[0] 
            flux[idx, :data[1].shape[0]] = data[1]
            idx += 1

        
    time = torch.Tensor(time)
    flux = torch.Tensor(flux)
    if have_labels:
        labels = pd.get_dummies(labels, dtype=float)
        labels = labels.to_numpy()
        labels = torch.Tensor(labels)
        print(flux.shape, time.shape, labels.shape)

        # can save tensors on device if needec:

        # torch.save(time, 'timetensor2.pt')
        # torch.save(flux, 'fluxtensor2.pt')
        # torch.save(labels, 'labelstensor2.pt')
        return time, flux, labels
    else:
        print(flux.shape, time.shape)
        return time, flux
    
if __name__ == '__main__':
    load_tess_qlp_data()
    # load_data("/Users/paul/Downloads/keplerq9v3/lightcurves", have_labels=True)