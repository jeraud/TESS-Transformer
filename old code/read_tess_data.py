from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import lightkurve as lk
import os
import Helper_functions.utils as utils
from scipy.ndimage import gaussian_filter1d
import torch



tess_data_dir = '/Users/paul/Desktop/UROP/tessv1/'
TLGC = '/Users/paul/Desktop/UROP/selected_lc/'
test_curve = '/Users/paul/Desktop/UROP/tessv1/ECLIPSE/hlsp_qlp_tess_ffi_s0014-0000000123316709_tess_v01_llc.fits'
targets = '/Users/paul/Desktop/UROP/tessv1/targets_qlp.csv'
targets2 = '/Users/paul/Desktop/UROP/tessv1/targets2.ecsv'
constants = '/Users/paul/Desktop/UROP/tessv1/CONSTANT/'
r_ceph = '/Users/paul/Desktop/UROP/tessv2_yeschen/RRLYR_CEPHEID'
DIR_LIGHT_CURVES = "/Users/paul/Downloads/keplerq9v3/lightcurves"
tessv2 = '/Users/paul/Desktop/UROP/tessv2_yeschen/'
def extract(curve_path):
    with fits.open(curve_path, mode="readonly") as hdulist:
        ticid = hdulist[0].header['TICID']
        # print(hdulist[0].header['TICID'])
        dont_exclude = [0,64,256,1024,2048,8192]
        # read time, sap flux, quality flag
        tess_bjds = hdulist[1].data['TIME']
        print(hdulist[0].header)
        sap_fluxes = hdulist[1].data['SAP_FLUX']
        qual_flags = hdulist[1].data['QUALITY']
        # remove flagged data
        where_gt0 = np.where(np.isin(qual_flags,dont_exclude))
        tess_bjds = tess_bjds[where_gt0]
        sap_fluxes = sap_fluxes[where_gt0]
        return (tess_bjds, sap_fluxes, ticid)




def extract_TGLC(curve_path):
    """
    Extract the time, fluxes from a fits file
    """
    with fits.open(curve_path, mode="readonly") as hdulist:
        #Flags to keep:
        dont_exclude = [0,64,256,1024,2048,8192]
        ticid = hdulist[0].header['TICID']
        # read time, sap flux, quality flag
        tess_bjds = hdulist[1].data['time']
        sap_fluxes = hdulist[1].data['aperture_flux']
        qual_flags = hdulist[1].data['TESS_flags']

        # remove flagged data
        where_no_flag = np.where(np.isin(qual_flags,dont_exclude))
        tess_bjds = tess_bjds[where_no_flag]
        sap_fluxes = sap_fluxes[where_no_flag]
        return (tess_bjds, sap_fluxes, ticid)

def get_tglc_lightcurves():
    TICdict = {}
    for fit in os.listdir(TLGC):
        path = '/Users/paul/Desktop/UROP/selected_lc/' + fit
        extracted = extract_TGLC(path)
        # print(extracted)
        TICdict[int(extracted[2])] = path
    return TICdict

MAX_LEN = 1171

def tess_kepler_combined():
    labels = []
    flux = np.zeros((4377 + len(os.listdir(DIR_LIGHT_CURVES)) + 1000, MAX_LEN))
    time = np.zeros((4377 + len(os.listdir(DIR_LIGHT_CURVES)) + 1000, MAX_LEN))
    df = pd.read_csv(targets)
    idx = 0
    for row in df.iterrows():
        curve = tess_data_dir + row[1].iloc[3]
        extracted = extract(curve)
        lc = lk.LightCurve({'time':extracted[0], 'flux':extracted[1]})
        lc = lc.head(MAX_LEN).remove_nans().remove_outliers(sigma=10)
        t = np.array(lc.time.value) 
        t = t + -1*t[0] + 0.0001
        f = np.array(lc.flux.value) - gaussian_filter1d(np.array(lc.flux.value), 61)
        labels.append(row[1].iloc[1])
        time[idx, :t.shape[0]] = t 
        flux[idx, :f.shape[0]] = (f - np.median(f)) / np.std(f)
        idx += 1
    constants = '/Users/paul/Desktop/UROP/tessv1/CONSTANT'
    filename_list = os.listdir(constants)
    for i, filename in enumerate(filename_list):
        try:
            df = utils.open_light_curve_csv(filename, constants)
            labels.append('CONSTANT')
            lc = lk.LightCurve({'time':df['time'], 'flux':df['flux']})
            lc = lc.head(MAX_LEN).remove_nans().remove_outliers(sigma=10)
            # .flatten(window_length=101)
            t = np.array(lc.time.value) 
            t = t + -1*t[0] + 0.0001
            f = np.array(lc.flux.value) - gaussian_filter1d(np.array(lc.flux.value), 61)
            # - np.mean(np.array(lc.flux.value))
            print(idx)
            # labels.append(row[1].iloc[1])
            time[idx, :t.shape[0]] = t 
            flux[idx, :f.shape[0]] = (f - np.median(f)) / np.std(f)
            idx += 1
        except:
            print(filename)
    filename_list = os.listdir(DIR_LIGHT_CURVES)
    for i, filename in enumerate(filename_list):
        # print(f"{i} of {len(filename_list)}")
        if filename.endswith(".txt"):
            df = utils.open_light_curve_csv(filename, DIR_LIGHT_CURVES)
            f = df['flux']
            # If less than n_max_obs observations, pad with zeros
            lc = lk.LightCurve({'time':df['time'], 'flux':df['flux']})
            lc = lc.head(MAX_LEN).remove_nans().remove_outliers(sigma=10)
            t = np.array(lc.time.value) 
            f = np.array(lc.flux.value) - gaussian_filter1d(np.array(lc.flux.value), 61)
            time[idx, :t.shape[0]] = t 
            flux[idx, :f.shape[0]] = (f - np.median(f)) / np.std(f)
            idx += 1
    ls = os.listdir(DIR_LIGHT_CURVES)
    df_ls = pd.DataFrame(ls, columns=['filename'])
    Klabels = pd.read_csv('/Users/paul/Downloads/keplerq9v3/targets.txt', header=None, names=['tic','class'], comment='#')
    Klabels = Klabels[Klabels['class']!='INSTRUMENT']
    
    #Order Klabels according to light curve files
    Klabels['filename'] = Klabels['tic'].apply('{:0>9}'.format) + '.txt'
    Klabels = df_ls.merge(Klabels,on='filename', how='left')[['tic','class']]
    
    Klabels_onehot = pd.get_dummies(Klabels['class']).to_numpy()
    flux = torch.Tensor(flux)
    time = torch.Tensor(time)
    labels = pd.get_dummies(labels)
    labels = labels.to_numpy()
    print(labels.shape, Klabels_onehot.shape)
    labels = np.vstack((labels, Klabels_onehot))
    labels = torch.Tensor(labels)
    torch.save(time, 'combotimetensor.pt')
    torch.save(flux, 'combofluxtensor.pt')
    torch.save(labels, 'combolabelstensor.pt')
    print(flux.shape, time.shape, labels.shape)
    # print(counter)
    return (time, flux, labels)


# tess_kepler_combined()


MAX_SEQ_LEN = 3917
def get_tess_data_TGLC():
    # max = 0
    classes = ['APERIODIC', 'CONSTANT', 'CONTACT_ROT', 'DSCT_BCEP', 'ECLIPSE', 'GDOR_SPB', 'RRLYR_CEPH', 'SOLARLIKE']
    # population_dict = {label:0 for label in classes}
    labels = []
    TICdict = get_tglc_lightcurves()
    # print(TICdict)
    # flux = np.zeros((5377, MAX_SEQ_LEN))
    # time = np.zeros((5377, MAX_SEQ_LEN))
    flux = np.zeros((5113, 1171))
    time = np.zeros((5113, 1171))
    df = pd.read_csv(targets)
    idx = 0
    for row in df.iterrows():
            print(idx)
            TIC = row[1].iloc[0]
            if TIC in TICdict:
                extracted = extract_TGLC(TICdict[int(TIC)])
                # remove nan flux values, outliers
                lc = lk.LightCurve({'time':extracted[0], 'flux':extracted[1]})
                lc = lc.remove_nans().remove_outliers(sigma=10).head(3513)
                t = np.array(lc.time.value) 
                num_bad = 0
                if t.shape[0] != 3513:
                    num_bad += 1
                    continue
                # t = t 
                # + -1*t[0] + 0.0001
                f = np.array(lc.flux.value) 
                # - gaussian_filter1d(np.array(lc.flux.value), 53)
                labels.append(row[1].iloc[1])
                #t = np.array(lc.time.value) 
            # t = t 
                f = np.array(lc.flux.value)
                binned_time = t[:1171 * 3].reshape(-1, 3).mean(axis=1)
                binned_flux = f[:1171 * 3].reshape(-1, 3).mean(axis=1)
                # - np.mean(np.array(lc.flux.value))
                print(idx)
                # labels.append(row[1].iloc[1])
                binned_time = binned_time + -1*binned_time[0] + 0.0001
                time[idx, :t.shape[0]] = binned_time
                binned_flux = binned_flux - gaussian_filter1d(binned_flux, 61)
                f = binned_flux
                flux[idx, :f.shape[0]] = (f - np.median(f)) / np.std(f)
                idx += 1
    filename_list = os.listdir(constants)
    print('bad!!!!!!!!', num_bad)
    for i, filename in enumerate(filename_list):
        try:
            df = utils.open_light_curve_csv(filename, constants)
            labels.append('CONSTANT')
            # population_dict['CONSTANT'] += 1
            lc = lk.LightCurve({'time':df['time'], 'flux':df['flux']})
            lc = lc.remove_nans().remove_outliers(sigma=10).head(3513)
            # .flatten(window_length=101)
            t = np.array(lc.time.value) 
            # t = t 
            f = np.array(lc.flux.value)
            binned_time = t[:1171 * 3].reshape(-1, 3).mean(axis=1)
    
    # Reshape and average the flux array
            binned_flux = f[:1171 * 3].reshape(-1, 3).mean(axis=1)
            # - np.mean(np.array(lc.flux.value))
            print(idx)
            # labels.append(row[1].iloc[1])
            binned_time = binned_time + -1*binned_time[0] + 0.0001
            time[idx, :t.shape[0]] = binned_time
            binned_flux = binned_flux - gaussian_filter1d(binned_flux, 61)
            f = binned_flux
            flux[idx, :f.shape[0]] = (f - np.median(f)) / np.std(f)
            idx += 1
        except:
            print(filename)
    flux = torch.Tensor(flux)
    time = torch.Tensor(time)
    labels = pd.get_dummies(labels)
    labels = labels.to_numpy()
    labels = torch.Tensor(labels)
    print(flux.shape, time.shape, labels.shape)
    # print(population_dict)
    torch.save(time, 'timetensorTLGClen1171.pt')
    torch.save(flux, 'fluxtensorTLGClen1171.pt')
    torch.save(labels, 'labelstensorTLGClen1171.pt')
    return (time, flux, labels)

# get_tess_data_TGLC()


def plot_curves(i):
    time = torch.load('/Users/paul/Desktop/UROP/tensors/qlptimetensor.pt')
    flux = torch.load('/Users/paul/Desktop/UROP/tensors/qlpfluxtensor.pt')
    time2 = torch.load('/Users/paul/Desktop/UROP/tensors/qlptimetensorRAW.pt')
    flux2 = torch.load('/Users/paul/Desktop/UROP/tensors/qlpfluxtensorRAW.pt')
    labels = torch.load('/Users/paul/Desktop/UROP/tensors/qlplabelstensor.pt')
    # labels2 = torch.load('qlplabelstensor.pt')
    # labels = torch.load('labelstensorrs.pt')
    lc = lk.LightCurve({'time':np.trim_zeros(np.array(time[i])), 'flux':np.trim_zeros(np.array(flux[i]))})
    lc2 = lk.LightCurve({'time':np.trim_zeros(np.array(time2[i])), 'flux':np.trim_zeros(np.array(flux2[i]))})
    # assert(labels[i] == labels2[i])
    lc.plot()
    plt.title('Example Preprocessed Light Curve, label: ' + str(labels[i]))
    plt.xlabel('Time')
    plt.ylabel('SAP Flux')
    plt.subplots_adjust(bottom=0.2)
    plt.savefig('/Users/paul/Desktop/UROP/tess-wavelets/qlp_processed.png')
    plt.close()
    lc2.plot()
    plt.title('Example Raw Light Curve, label: ' + str(labels[i]))
    plt.xlabel('Time')
    plt.ylabel('SAP Flux')
    plt.subplots_adjust(bottom=0.2)
    plt.savefig('/Users/paul/Desktop/UROP/tess-wavelets/qlp_raw.png')
    plt.close()
    print(labels[0])
plot_curves(5000)

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
    flux = torch.Tensor(flux)
    labels = pd.get_dummies(labels)
    labels = labels.to_numpy()
    labels = torch.Tensor(labels)
    print(flux.shape, time.shape, labels.shape)
    torch.save(time, 'timetensor2.pt')
    torch.save(flux, 'fluxtensor2.pt')
    torch.save(labels, 'labelstensor2.pt')
    return flux, labels, time

# get_large_val_set('/Users/paul/Desktop/UROP/tessv2_yeschen/')


YECURVES = '/Users/paul/Desktop/pdo/users/yeschen/lcs_p1/qlp/'
def run_on_yeschen():
    flux = np.zeros((9536, 1171))
    time = np.zeros((9536, 1171))
    idx = 0
    max = 0
    # print(len([f for f in os.listdir(YECURVES) if os.path.isfile(os.path.join(YECURVES, f))]))
    for filename in os.listdir(YECURVES):
        try:

            file_path = YECURVES + filename
            df = pd.read_csv(file_path, header=None, index_col=None, names=['raw','bjds','cadences','errors','qflags','lc_length'], low_memory=False)
            df = df[1:]
            flags = pd.to_numeric(df.qflags)
            fluxlc = pd.to_numeric(df.where(flags==0).raw)
            # print(df.bjds)
            # print(pd.to_numeric(df.bjds[1:]))
            # # print(df['bjds'])
            lc = lk.LightCurve({'time':pd.to_numeric(df.bjds, errors='coerce'), 'flux':fluxlc})
            lc = lc.remove_nans().remove_outliers(sigma=10)
            t = np.array(lc.time.value) 
            t = t + -1*t[0] + 0.0001
            f = np.array(lc.flux.value) - gaussian_filter1d(np.array(lc.flux.value), 61) 
            # rand_idx = np.random.choice(f.size, 1171, replace=False)
            # flux[idx, :] = f[rand_idx]
            # time[idx, :] = t[rand_idx]
            # - gaussian_filter1d(np.array(lc.flux.value), 61) 
            # size = f.size
            # print(size)
            # if size > max:
            #     max = size
            # time[idx, :t.shape[0]] = t 
            # flux[idx, :f.shape[0]] = (f - np.median(f)) / np.std(f)
            idx += 1
            print(idx)
            # lc.plot()
            # plt.title('YESCHEN Example Light Curve')
            # plt.xlabel('Time')
            # plt.ylabel('SAP Flux')
            # plt.savefig('/Users/paul/Desktop/UROP/tess-wavelets/test_curveYoshi.png')
            # plt.close()
            # return None
            # print(max)
        except:
            flux = np.delete(flux, -1, 0)
            time = np.delete(time, -1, 0)
            print(filename)
    time = torch.Tensor(time)
    flux = torch.Tensor(flux)
    torch.save(time, 'yeschentime.pt')
    torch.save(flux, 'yeschenflux.pt')
# run_on_yeschen()

def load_tess_qlp_data():
    """
    load labeled tess qlp data into tensors 
    """
    labels = []
    # flux = np.zeros((5933, MAX_LEN))
    # time = np.zeros((5933, MAX_LEN))
    # df = pd.read_csv(targets)
    # idx = 0
    # ticid_set = set()
    # for row in df.iterrows():
    #     label = row[1].iloc[1]
    #     curve = tess_data_dir + row[1].iloc[3]
    #     extracted = extract(curve)
    #     # print(extracted[2])
    #     ticid_set.add(extracted[2])
    num_repeat = 0
    df = pd.read_csv(targets2)
    for row in df.iterrows():
        # if row[1].iloc[0] not in ticid_set:
            num_repeat += 1
            curve = row[1].iloc[3]
            extracted = extract(os.path.join(tess_data_dir,curve))
            print(extracted[1])
        
        # extracted = extract(curve)
    #     if extracted[2] in ticid_set:
    #         num_repeat+=1
    # print(num_repeat)
        # lc = lk.LightCurve({'time':extracted[0], 'flux':extracted[1]})
        # lc = lc.remove_nans().remove_outliers(sigma=10).head(MAX_LEN)
        # t = np.array(lc.time.value) 
        # t = t + -1*t[0] + 0.0001
        # f = np.array(lc.flux.value) - gaussian_filter1d(np.array(lc.flux.value), 61)
        # # print(f.shape)
        # # f = f.reshape(-1, 2).mean(axis=1)
        # # t = t.reshape(-1, 2).mean(axis=1)
        # labels.append(label)
        # time[idx, :t.shape[0]] = t 
        # flux[idx, :f.shape[0]] = (f - np.median(f)) / np.std(f)
        # idx += 1
    
    # for filename in os.listdir(r_ceph):
    #     if filename.startswith("hlsp_qlp") and filename.endswith('.fits'):
    #                 extracted = extract(os.path.join(r_ceph,filename))
    #                 lc = lk.LightCurve({'time':extracted[0], 'flux':extracted[1]})
    #                 lc = lc.remove_nans().remove_outliers(sigma=10).head(MAX_LEN)
    #                 t = np.array(lc.time.value) 
    #                 t = t + -1*t[0] + 0.0001
    #                 f = np.array(lc.flux.value) - gaussian_filter1d(np.array(lc.flux.value), 61)
    #                 print(idx)
    #                 # f = f.reshape(-1, 2).mean(axis=1)
    #                 # t = t.reshape(-1, 2).mean(axis=1)
    #                 labels.append('RRLYR_CEPHEID')
    #                 time[idx, :t.shape[0]] = t 
    #                 flux[idx, :f.shape[0]] = (f - np.median(f)) / np.std(f)
    #                 idx += 1
        
    # constants = '/Users/paul/Desktop/UROP/tessv1/CONSTANT/'
    # filename_list = os.listdir(constants)   
    # for i, filename in enumerate(filename_list):
    #     try:
    #         df = utils.open_light_curve_csv(filename, constants)
    #         labels.append('CONSTANT')
    #         lc = lk.LightCurve({'time':df['time'], 'flux':df['flux']})
    #         lc = lc.remove_nans().remove_outliers(sigma=10).head(MAX_LEN)
    #         # .flatten(window_length=101)
    #         t = np.array(lc.time.value) 
    #         t = t + -1*t[0] + 0.0001
    #         f = np.array(lc.flux.value) - gaussian_filter1d(np.array(lc.flux.value), 61)
    #         # - np.mean(np.array(lc.flux.value))
    #         # f = f.reshape(-1, 2).mean(axis=1)
    #         # t = t.reshape(-1, 2).mean(axis=1)
    #         print(idx)
    #         # labels.append(row[1].iloc[1])
    #         time[idx, :t.shape[0]] = t 
    #         flux[idx, :f.shape[0]] = (f - np.median(f)) / np.std(f)
    #         idx += 1
    #     except:
    #         print(filename)
    # flux = torch.Tensor(flux)
    # time = torch.Tensor(time)
    # labels = pd.get_dummies(labels, dtype=float)
    # labels = labels.to_numpy()
    # labels = torch.tensor(labels)
    # print(labels[0])
    # torch.save(time, 'qlptimetensorRAWR.pt')
    # torch.save(flux, 'qlpfluxtensorRAW.pt')
    # torch.save(labels, 'qlplabelstensorRAW.pt')
    # print(flux.shape, time.shape, labels.shape)
    print(num_repeat)
    # return ticid_set

# load_tess_qlp_data()
def get_tessv2():
    ticid_set = load_tess_qlp_data()
    num_repeat = 0
    df = pd.read_csv(os.path.join(tessv2, 'targets.ecsv'))
    for row in df.iterrows():
        ticid = row[1].iloc[2]
        if ticid not in ticid_set:
            if (row[1].iloc[4] == 'CONSTANT'):
                num_repeat += 1
    # for label in os.listdir(tessv2):
    #     print(label)
    #     for curve in os.listdir(os.path.join(tessv2,label)):
    #         if curve.startswith("hlsp_qlp") and curve.endswith('.fits'):
    #             extracted = extract(os.path.join(tessv2, label,curve))
    #             if extracted[2] not in ticid_set:
    #                 num_repeat += 1
    print(num_repeat)

# get_tessv2()


def get_population():
    classes = ['APERIODIC', 'CONSTANT', 'CONTACT_ROT', 'DSCT_BCEP', 'ECLIPSE', 'GDOR_SPB', 'RRLYR_CEPH', 'SOLARLIKE']
    population_dict = {label:0 for label in classes}
    labels = torch.load('qlplabelstensor.pt')
    population_dict = {label:0 for label in classes}
    # print(labels[0])
    for label in labels:
        nonzero_index = torch.nonzero(label, as_tuple=True)[0]
        population_dict[classes[nonzero_index]] += 1
    
    print(population_dict)
# get_population()