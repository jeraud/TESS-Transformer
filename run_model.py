import sys, os
sys.path.append('./')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset

from torchmetrics.classification import MulticlassConfusionMatrix

from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import Callback

import utils

from light_curve_classifier import LightCurveClassifier

class AccuracyLogger(Callback):
    def __init__(self):
        self.val_acc = []

    def on_validation_end(self, trainer, pl_module):
        self.val_acc.append(trainer.callback_metrics['val_acc'].item())

def main():
        DIR_LIGHT_CURVES = "/Users/paul/Downloads/keplerq9v3/lightcurves"
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
            # print(f"{i} of {len(filename_list)}")
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
        for i in range(flux.shape[0]):
            flux[i] = (flux[i] - np.median(flux[i])) / np.std(flux[i])



        # plot example light curve
        # plt.figure()
        # plt.scatter(time[0], flux[0])
        # plt.savefig('temp.png')
        # plt.close()

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
 
        class_weights =  total / labels.sum(axis=0)

        # initialize classifier
        mlce = LightCurveClassifier(lr=6e-4, transformer_kwargs={"emb":64, "heads":8, "layers":3, "dropout_p":0.3, "hidden":256},optimizer_kwargs={}, num_classes=8, class_weights=class_weights)
        
        nobjects = flux.shape[0]
        val_fraction = 0.1
        test_fraction = 0.19
        batch_size = 32
        n_samples_test = int(test_fraction * nobjects)
        n_samples_val = int(val_fraction * nobjects)

        dataset = TensorDataset(flux, time, labels, mask)

        train_val_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=n_samples_test,random_state=42, shuffle=True, stratify=labels)
        train_idx, validation_idx = train_test_split(train_val_idx, test_size=n_samples_val,random_state=42, shuffle=True, stratify=labels[train_val_idx])

        dataset_train = Subset(dataset, train_idx)
        dataset_val = Subset(dataset, validation_idx)
        dataset_test = Subset(dataset, test_idx)


        train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=True)
        val_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=False)
        test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=False)

        print(torch.cuda.is_available())

        # Initialize Callbacks
        early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_acc", stopping_threshold=0.93)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_acc", save_top_k=1, mode="max")
        accuracy_logger = AccuracyLogger()

        trainer = pl.Trainer(max_epochs=150,
                            min_epochs=100,
                             accelerator='auto',
                             devices='auto',
                            #  num_nodes=1,
                            #  strategy= 'ddp',
                             callbacks=[early_stop_callback,checkpoint_callback, accuracy_logger])
        print('starting training')
        trainer.fit(model=mlce, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.save_checkpoint("cl_model.ckpt")

        print(trainer.test(dataloaders=test_loader))

        # plot validation accuracy
        epochs = range(1, len(accuracy_logger.val_acc) + 1)
        plt.plot(epochs, accuracy_logger.val_acc, 'b', label='Validation accuracy')
        plt.title('Validation Accuracy During Training')
        plt.xlabel('Epochs')
        plt.ylabel('Validation Accuracy')
        plt.savefig('val_acc.png')
        plt.close()

        # make confusion matrix
        metric = mlce.conf_matrix.to('cuda')
        fig_, ax_ = metric.plot(labels=['APERIODIC', 'CONSTANT', 'CONTACT_ROT', 'DSCT_BCEP', 'ECLIPSE', 'GDOR_SPB', 'RRLYR_CEPH', 'SOLARLIKE'])
        plt.savefig('confusion_matrix.png')
        plt.close()

        # plot misclassifications
        names = ['APERIODIC', 'CONSTANT', 'CONTACT_ROT', 'DSCT_BCEP', 'ECLIPSE', 'GDOR_SPB', 'RRLYR_CEPH', 'SOLARLIKE']
        for missede in enumerate(mlce.testmc):
            missed = missede[1]
            print(missed[0].shape, missed[1].shape)
            data = (missed[1], missed[0])
            pred = missed[2].item()
            actual = torch.nonzero(missed[3] == 1).squeeze().item()
            plt.plot(data[0].cpu(), data[1].cpu())
            plt.title('predicted ' + names[pred] + ', was ' + names[actual])
            plt.xlabel('time')
            plt.ylabel('flux')
            file_path = os.path.join('missclassified_curves', 'missed_curve' + str(missede[0]))
            plt.savefig(file_path)
            plt.close()
        

if __name__ == '__main__':
     main()