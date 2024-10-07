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

import lightkurve as lk
import lightkurve.periodogram as periodogram

import utils

from light_curve_classifier import LightCurveClassifier

from load_data import load_data


# from read_tess_data import get_tess_data
class AccuracyLogger(Callback):
    """
    callback function to track validaiton accuracy during training
    """
    def __init__(self):
        self.val_acc = []

    def on_validation_end(self, trainer, pl_module):
        self.val_acc.append(trainer.callback_metrics['val_acc'].item())

def main(data, load_from_device = True):
        """
        Trains the model

        @args:
            load_from_device(bool): Whether or not the training set time, flux, label 
                                    tensors have been already loaded and saved on device.
                                    recommended to save tensors to device, load them here,
                                    saves time not having to load data every time you want to train,
                                    is also easier for training on cluster.
            data(tuple[string] or tuple[tensors]): tuple of (time, flux, labels) where each is the string of the 
                                relative path to the tensors, ie ('timetensor.pt', 'fluxtensor.pt', 'labelstensor.pt')
                                if load_from_device = True, otherwise tuple of pytorch tensors (time, flux, labels)
        """
        pl.seed_everything(42, workers=True)
        # if load from device, load here
        if load_from_device:
            # time.shape = flux.shape = [n,T] ; labels.shape = [n,8]
            time, flux, labels = torch.load(data[0]), torch.load(data[1]), torch.load(data[2])

        else:
             time, flux, labels = data

                # Get class populations for weighted loss

        total = labels.sum()
 
        class_weights =  total / labels.sum(axis=0)

        # initialize classifier
        mlce = LightCurveClassifier(lr=1e-4, transformer_kwargs={"emb":64, "heads":8, "layers":3, "dropout_p":0.2, "hidden":256},optimizer_kwargs={}, num_classes=8, class_weights=class_weights)
        
        # train, validate, test splits
        nobjects = flux.shape[0]
        val_fraction = 0.1
        test_fraction = 0.2
        batch_size = 32
        n_samples_test = int(test_fraction * nobjects)
        n_samples_val = int(val_fraction * nobjects)

        dataset = TensorDataset(flux, time, labels)

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
        # model checkpoint: saves top k validation accuraccy models
        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_acc", save_top_k=5, mode="max", filename='{epoch:02d}-{val_acc:.2f}')
        # log validation accuraccy for plot
        accuracy_logger = AccuracyLogger()

        #initialize lightning trainer
        # uncomment lines for cluster training
        trainer = pl.Trainer(max_epochs=300,
                            min_epochs=50,
                            #  accelerator='auto',
                            #  devices='auto',
                            # #  num_nodes=1,
                            #  strategy= 'ddp',
                             callbacks=[checkpoint_callback, accuracy_logger])
        
        print('starting training')

        trainer.fit(model=mlce, train_dataloaders=train_loader, val_dataloaders=val_loader)

        top_k_checkpoints = checkpoint_callback.best_k_models.keys()
        best_acc = 0.0
        best_loss = 0.0
        best_model_path = None
        best = None
        bModel = None
        
        # find saved model that generalizes the best
        for checkpoint_path in top_k_checkpoints:
            model = LightCurveClassifier.load_from_checkpoint(checkpoint_path)
            trainer = pl.Trainer()
            test = trainer.test(model, test_loader)
            test_acc = test[0]['test_acc_epoch']
            test_loss = test[0]['test_loss_epoch']
            if (test_acc > best_acc) or (test_acc == best_acc and test_loss < best_loss):
                best_acc = test_acc
                best_loss = test_loss
                best_model_path = checkpoint_path
                best = trainer
                bModel = model
            
            
 
        print(f'Best Model Path: {best_model_path} with Test Accuracy: {best_acc}')

        # save model weights 
        best.save_checkpoint("cl_model_" + str(best_acc) + ".ckpt")
        
        # plot validation accuracy
        epochs = range(1, len(accuracy_logger.val_acc) + 1)
        plt.plot(epochs, accuracy_logger.val_acc, 'b', label='Validation accuracy')
        plt.title('Validation Accuracy During Training')
        plt.xlabel('Epochs')
        plt.ylabel('Validation Accuracy')
        plt.savefig('val_acc_' + str(best_acc) + '.png')
        plt.close()

        # make confusion matrix
        metric = bModel.conf_matrix.to('cuda')
        fig_, ax_ = metric.plot(labels=['APERIODIC', 'CONSTANT', 'CONTACT_ROT', 'DSCT_BCEP', 'ECLIPSE', 'GDOR_SPB', 'RRLYR_CEPH', 'SOLARLIKE'])
  
        plt.savefig('confusion_matrix_' + str(best_acc) + '.png')
        plt.close()

        #plot missclassified curves duriing testing
        names = ['APERIODIC', 'CONSTANT', 'CONTACT_ROT', 'DSCT_BCEP', 'ECLIPSE', 'GDOR_SPB', 'RRLYR_CEPH', 'SOLARLIKE']
        for missede in enumerate(bModel.testmc):
            missed = missede[1]
            data = (missed[1], missed[0])
            pred = missed[2].item()
            actual = torch.nonzero(missed[3] == 1).squeeze().item()
            plt.plot(data[0].cpu(), data[1].cpu())
            plt.title('light curve ' + str(missede[0]) + ': '+ 'predicted ' + names[pred] + ', was ' + names[actual])
            plt.xlabel('time')
            plt.ylabel('flux')
            file_path = os.path.join('missclassified_curves', 'missed_curve_lc' + str(missede[0]))
            plt.savefig(file_path)
            plt.close()
        

if __name__ == '__main__':
     main(load_from_device=True, data = ('combotimetensor.pt', 'combofluxtensor.pt', 'combolabelstensor.pt'))