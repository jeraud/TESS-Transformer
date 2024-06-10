import sys, os
sys.path.append('./')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
from lightning.pytorch.loggers import WandbLogger

from torchmetrics.classification import MulticlassConfusionMatrix

from sklearn.model_selection import train_test_split

import wandb

import utils

# from wave_scat_conformer import WaveScatFormer
from light_curve_classifier import LightCurveClassifier
# def text_file_to_matrix(file_path, delimiter=','):
#     with open(file_path, 'r') as file:
#         # Read all lines from the file
#         lines = file.readlines()
        
#         # Split each line into values and convert to float
#         matrix = [list(map(float, line.strip().split(delimiter))) for line in lines]
        
#     return torch.tensor(matrix)

# input = text_file_to_matrix('/Users/paul/Downloads/keplerq9v3/APERIODIC/001026895.txt', delimiter=None)
# print(input.shape)
# mask =torch.isnan(input).any(dim=1)
# input = input[~mask]
# former = WaveScatFormer(512, 4)
# print(former(input.unsqueeze(0)))
def main():
        DIR_LIGHT_CURVES = "/Users/paul/Downloads/keplerq9v3/lightcurves/"
        pl.seed_everything(42, workers=True)

        #### Read in light curves for training

        n_max_obs = 1350

        # Open all light curves and save as list. Pad observations with zeros up to n_max_obs.
        flux = np.zeros((len(os.listdir(DIR_LIGHT_CURVES)), n_max_obs))
        time = np.zeros((len(os.listdir(DIR_LIGHT_CURVES)), n_max_obs))
        mask = np.zeros((len(os.listdir(DIR_LIGHT_CURVES)), n_max_obs), dtype=bool)

        filename_list = os.listdir(DIR_LIGHT_CURVES)
        idx = 0
        for i, filename in enumerate(filename_list):
            print(f"{i} of {len(filename_list)}")
            if filename.endswith(".txt"):
                df = utils.open_light_curve_csv(filename, DIR_LIGHT_CURVES)
                # If less than n_max_obs observations, pad with zeros
                if len(df['flux']) < n_max_obs:
                    flux[idx, :len(df['flux'])] = df['flux']
                    time[idx, :len(df['flux'])] = df['time']
                    mask[idx, :len(df['flux'])] = 1
                # Otherwise, randomly select n_max_obs observations
                else:
                    rand_idx = np.random.choice(len(df['flux']), n_max_obs, replace=False)
                    flux[idx, :] = df['flux'][rand_idx]
                    time[idx, :] = df['time'][rand_idx]
                    mask[idx, :] = 1
                idx += 1


        # Standardize fluxes
        # flux_means = np.median(flux, axis=1)
        # flux_stds = np.std(flux, axis=1)
        for i in range(flux.shape[0]):
            flux[i] = (flux[i] - np.median(flux[i])) / np.std(flux[i])



        # plot example light curve
        plt.figure()
        plt.scatter(time[0], flux[0])
        plt.savefig('temp.png')

        # Convert to torch tensors
        flux = torch.Tensor(flux)
        time = torch.Tensor(time)
        mask = torch.Tensor(mask).to(torch.bool)

        # Read in light curve ordering
        ls = os.listdir(DIR_LIGHT_CURVES)
        df_ls = pd.DataFrame(ls, columns=['filename'])

        # Read labels
        labels = pd.read_csv('/Users/paul/Downloads/keplerq9v3/targets.txt', header=None, names=['tic','class'], comment='#')
        labels = labels[labels['class']!='INSTRUMENT']
        
        #Order labels according to light curve files
        labels['filename'] = labels['tic'].apply('{:0>9}'.format) + '.txt'
        labels = df_ls.merge(labels,on='filename', how='left')[['tic','class']]
        
        labels_onehot = pd.get_dummies(labels['class']).to_numpy()
        labels = torch.Tensor(labels_onehot)

        # Get class populations for weighted loss
        total = labels.sum()
        # print(total)
        # print(labels.sum(axis=0))
        class_weights =  total / labels.sum(axis=0)
        # print('w',class_weights)
        # print(flux.shape)
        # former = LightCurveClassifier(512,5)
        # print(LightCurveClassifier(flux))
        # Set up Transformer classifier
        mlce = LightCurveClassifier(lr=6e-4, transformer_kwargs={"emb":64, "heads":4, "layers":4, "dropout_p":0.05, "hidden":256},optimizer_kwargs={}, num_classes=8, class_weights=class_weights)
        
        nobjects = flux.shape[0]
        val_fraction = 0.05
        test_fraction = 0.05
        batch_size = 32
        n_samples_test = int(test_fraction * nobjects)
        n_samples_val = int(val_fraction * nobjects)

        dataset = TensorDataset(flux, time, labels, mask)
        # print(dataset)

        train_val_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=n_samples_test,random_state=42, shuffle=True, stratify=labels)
        train_idx, validation_idx = train_test_split(train_val_idx, test_size=n_samples_val,random_state=42, shuffle=True, stratify=labels[train_val_idx])

        dataset_train = Subset(dataset, train_idx)
        dataset_val = Subset(dataset, validation_idx)
        dataset_test = Subset(dataset, test_idx)


        train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False)
        test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False)
        # print(former(flux[0].unsqueeze(0)))
        # Non-stratified sampler
        dataset_train, dataset_val = random_split(dataset, [flux.shape[0] - n_samples_val, n_samples_val])
        train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False)

        print(torch.cuda.is_available())

        # Initialize wandb logger
        # wandb_logger = WandbLogger(project='wandb-lightning', job_type='train')

        # Initialize Callbacks
        early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_acc", patience=10)
        checkpoint_callback = pl.callbacks.ModelCheckpoint()

        trainer = pl.Trainer(max_epochs=50,
                             accelerator='gpu',
                             # logger=wandb_logger,
                             callbacks=[early_stop_callback,checkpoint_callback,])

        trainer.fit(model=mlce, train_dataloaders=train_loader, val_dataloaders=val_loader)

        trainer.save_checkpoint("cl_model.ckpt")

        # Close wandb run
        # wandb.finish()

        # print(mlce.test_step(test_loader,test_idx))
        #Make confusion matrix
        preds = mlce(flux[test_idx], time[test_idx])
        metric = MulticlassConfusionMatrix(num_classes=8)
        labe = labels_onehot[test_idx]
        # metric(preds,labe)
        fig_, ax_ = metric.plot(labels=['APERIODIC', 'CONSTANT', 'CONTACT_ROT', 'DSCT_BCEP', 'ECLIPSE', 'GDOR_SPB', 'RRLYR_CEPH', 'SOLARLIKE'])
        plt.savefig('confusion_matrix.png')
        plt.close()
        

if __name__ == '__main__':
     main()