from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import lightkurve as lk
import os
import utils
from scipy.ndimage import gaussian_filter1d
import torch



tess_data_dir = '/Users/paul/Desktop/UROP/tessv1/'
test_curve = '/Users/paul/Desktop/UROP/tessv1/ECLIPSE/hlsp_qlp_tess_ffi_s0014-0000000123316709_tess_v01_llc.fits'
targets = '/Users/paul/Desktop/UROP/tessv1/targets_qlp.csv'
constants = '/Users/paul/Desktop/UROP/tessv1/CONSTANT/'
r_ceph = '/Users/paul/Desktop/UROP/tessv2_yeschen/RRLYR_CEPHEID'

def extract(curve_path):
    with fits.open(curve_path, mode="readonly") as hdulist:
        dont_exclude = [0,64,256,1024,2048,8192]
        # read time, sap flux, quality flag
        tess_bjds = hdulist[1].data['TIME']
        sap_fluxes = hdulist[1].data['SAP_FLUX']
        qual_flags = hdulist[1].data['QUALITY']
        # remove flagged data
        where_gt0 = np.where(np.isin(qual_flags,dont_exclude))
        tess_bjds = tess_bjds[where_gt0]
        sap_fluxes = sap_fluxes[where_gt0]
        return (tess_bjds, sap_fluxes)

# ex = extract(test_curve)
# lc = lk.LightCurve({'time':ex[0], 'flux':ex[1]})
# lc = lc.remove_nans().remove_outliers(sigma=10)
# t = np.array(lc.time.value) 
# t = t + -1*t[0] + 0.0001
# f = np.array(lc.flux.value) - gaussian_filter1d(np.array(lc.flux.value), 61)
# lc = lk.LightCurve({'time':t, 'flux':f})
# create_plot(lc)
# power_spec = lc.to_periodogram(method='lombscargle')
# power_spec.plot()
# plt.savefig('/Users/paul/Desktop/UROP/tess-wavelets/test_curve2.png')
# plt.close()


MAX_SEQ_LEN = 1171
def get_tess_data():
    labels = []
    # flux = np.zeros((5377, MAX_SEQ_LEN))
    # time = np.zeros((5377, MAX_SEQ_LEN))
    flux = np.zeros((5933, MAX_SEQ_LEN))
    time = np.zeros((5933, MAX_SEQ_LEN))
    df = pd.read_csv(targets)
    idx = 0
    for row in df.iterrows():
            curve = tess_data_dir + row[1].iloc[3]
            extracted = extract(curve)
            # remove nan flux values, outliers
            lc = lk.LightCurve({'time':extracted[0], 'flux':extracted[1]})
            lc = lc.remove_nans().remove_outliers(sigma=10)
            # .flatten(window_length=101)
            t = np.array(lc.time.value) 
            t = t + -1*t[0] + 0.0001
            f = np.array(lc.flux.value) - gaussian_filter1d(np.array(lc.flux.value), 53)
            # - np.mean(np.array(lc.flux.value))
            print(idx)
            labels.append(row[1].iloc[1])
            time[idx, :t.shape[0]] = t 
            flux[idx, :f.shape[0]] = (f - np.median(f)) / np.std(f)
            idx += 1
    filename_list = os.listdir(constants)
    for i, filename in enumerate(filename_list):
        try:
            df = utils.open_light_curve_csv(filename, constants)
            labels.append('CONSTANT')
            lc = lk.LightCurve({'time':df['time'], 'flux':df['flux']})
            lc = lc.head(MAX_SEQ_LEN).remove_nans().remove_outliers(sigma=10)
            # .flatten(window_length=101)
            t = np.array(lc.time.value) 
            t = t + -1*t[0] + 0.0001
            f = np.array(lc.flux.value) - gaussian_filter1d(np.array(lc.flux.value), 53)
            # - np.mean(np.array(lc.flux.value))
            print(idx)
            # labels.append(row[1].iloc[1])
            time[idx, :t.shape[0]] = t 
            flux[idx, :f.shape[0]] = (f - np.median(f)) / np.std(f)
            idx += 1
        except:
            print(filename)
    for filename in os.listdir(r_ceph):
        extracted = extract(os.path.join(r_ceph,filename))
        lc = lk.LightCurve({'time':extracted[0], 'flux':extracted[1]})
        lc = lc.head(1171).remove_nans().remove_outliers(sigma=10)
        # .flatten(window_length=101)
        t = np.array(lc.time.value) 
        t = t + -1*t[0] + 0.0001
        f = np.array(lc.flux.value) - gaussian_filter1d(np.array(lc.flux.value), 53)
            # - np.mean(np.array(lc.flux.value))
        print(idx)
        # labels.append(row[1].iloc[1])
        time[idx, :t.shape[0]] = t 
        flux[idx, :f.shape[0]] = (f - np.median(f)) / np.std(f)
        # idx += 1
        labels.append('RRLYR_CEPHEID')
        # time[idx, :t.shape[0]] = t 
        # flux[idx, :f.shape[0]] = (f - np.median(f)) / np.std(f)
        idx += 1
    flux = torch.Tensor(flux)
    time = torch.Tensor(time)
    labels = pd.get_dummies(labels)
    labels = labels.to_numpy()
    labels = torch.Tensor(labels)
    print(flux.shape, time.shape, labels.shape)
    torch.save(time, 'timetensor45.pt')
    torch.save(flux, 'fluxtensor45.pt')
    torch.save(labels, 'labelstensor45.pt')
    return (time, flux, labels)

# get_tess_data()
def plot_curves():
    time = torch.load('timetensorrs.pt')
    flux = torch.load('fluxtensorrs.pt')
    labels = torch.load('labelstensorrs.pt')
    lc = lk.LightCurve({'time':np.trim_zeros(np.array(time[0])), 'flux':np.trim_zeros(np.array(flux[0]))})
    lc.plot()
    plt.title('Example Light Curve')
    plt.xlabel('Time')
    plt.ylabel('SAP Flux')
    plt.savefig('/Users/paul/Desktop/UROP/tess-wavelets/test_curve4.png')
    plt.close()
    print(labels[0])
# plot_curves()

def get_large_val_set(dir):
    labels = []
    flux = np.zeros((5910, 1191))
    time = np.zeros((5910, 1191))
    max_len = 0
    idx = 0
    constants = '/Users/paul/Desktop/UROP/tessv2_yeschen/CONSTANT'
    for item in os.listdir(dir):
        item_path = os.path.join(dir, item)
        if os.path.isdir(item_path) and item != "CONSTANT":
            for filename in os.listdir(item_path):
                if filename.startswith("hlsp_qlp") and filename.endswith('.fits'):
                    extracted = extract(os.path.join(item_path,filename))
                    lc = lk.LightCurve({'time':extracted[0], 'flux':extracted[1]})
                    lc = lc.remove_nans().remove_outliers(sigma=10)
                    t = np.array(lc.time.value) 
                    t = t + -1*t[0] + 0.0001
                    f = np.array(lc.flux.value) - gaussian_filter1d(np.array(lc.flux.value), 61)
                    print(idx)
                    labels.append(item)
                    time[idx, :t.shape[0]] = t 
                    flux[idx, :f.shape[0]] = (f - np.median(f)) / np.std(f)
                    idx += 1
    filename_list = os.listdir(constants)
    for i, filename in enumerate(filename_list):
        df = utils.open_light_curve_csv(filename, constants)
        labels.append('CONSTANT')
        lc = lk.LightCurve({'time':df['time'], 'flux':df['flux']})
        lc = lc.head(1191).remove_nans().remove_outliers(sigma=10)
        t = np.array(lc.time.value) 
        t = t + -1*t[0] + 0.0001
        f = np.array(lc.flux.value) - gaussian_filter1d(np.array(lc.flux.value), 61)
        time[idx, :t.shape[0]] = t 
        flux[idx, :f.shape[0]] = (f - np.median(f)) / np.std(f)
        idx += 1
    time = torch.Tensor(time)
    flux = torch.tensor(flux)
    labels = pd.get_dummies(labels)
    labels = labels.to_numpy()
    labels = torch.Tensor(labels)
    print(flux.shape, time.shape, labels.shape)
    torch.save(time, 'timetensor2.pt')
    torch.save(flux, 'fluxtensor2.pt')
    torch.save(labels, 'labelstensor2.pt')
    return flux, labels, time

# get_large_val_set('/Users/paul/Desktop/UROP/tessv2_yeschen/')



