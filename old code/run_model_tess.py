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

import helper_functions.utils as utils

from Model.light_curve_classifier import LightCurveClassifier


# from read_tess_data import get_tess_data
class AccuracyLogger(Callback):
    def __init__(self):
        self.val_acc = []

    def on_validation_end(self, trainer, pl_module):
        self.val_acc.append(trainer.callback_metrics['val_acc'].item())

def main():
        pl.seed_everything(42, workers=True)
        # dat = get_tess_data()
        # time, flux, labels = dat
        time = torch.load('qlptimetensor.pt')
        flux = torch.load('qlpfluxtensor.pt')
        labels = torch.load('qlplabelstensor.pt')
        print(time.shape, flux.shape, labels.shape)
        # Get class populations for weighted loss
        total = labels.sum()
 
        class_weights =  total / labels.sum(axis=0)
        print(class_weights)
        # class_weights = torch.tensor([10,1])

        # initialize classifier
        mlce = LightCurveClassifier(lr=1e-4, transformer_kwargs={"emb":64, "heads":8, "layers":3, "dropout_p":0.1, "hidden":256},optimizer_kwargs={}, num_classes=8, class_weights=class_weights)
        
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
        # early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_acc", stopping_threshold=0.89)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_acc", save_top_k=19, mode="max", filename='{epoch:02d}-{val_acc:.2f}')
        accuracy_logger = AccuracyLogger()

        trainer = pl.Trainer(max_epochs=3000,
                            min_epochs=500,
                             accelerator='auto',
                             devices='auto',
                             strategy='auto',
                            #  num_nodes=1,
                            #  strategy= 'ddp',
                             callbacks=[
                                 checkpoint_callback, 
                                 accuracy_logger])
        # print(pl.utilities.model_summary.summarize(mlce, max_depth = 4))
        print('starting training')
        trainer.fit(model=mlce, train_dataloaders=train_loader, val_dataloaders=val_loader)
       
       
        top_k_checkpoints = checkpoint_callback.best_k_models.keys()

        best_acc = 0.0
        best_loss = 0.0
        best_model_path = None
        best = None
        bModel = None
        
        # Test each of the top 5 models
        # print('here',top_k_checkpoints)
        for checkpoint_path in top_k_checkpoints:
            # print('here')
            model = LightCurveClassifier.load_from_checkpoint(checkpoint_path)
            trainer = pl.Trainer()
            test = trainer.test(model, test_loader)
            # print(test)
            test_acc = test[0]['test_acc_epoch']
            test_loss = test[0]['test_loss_epoch']
            # print(f'Checkpoint {checkpoint_path} Test Accuracy: {test_acc}')
            if (test_acc > best_acc) or (test_acc == best_acc and test_loss < best_loss):
                best_acc = test_acc
                best_loss = test_loss
                best_model_path = checkpoint_path
                best = trainer
                bModel = model
            
            
        # print('done testing')
        # Save the best model
        print(f'Best Model Path: {best_model_path} with Test Accuracy: {best_acc}')
        # best_model = LightCurveClassifier.load_from_checkpoint(best_model_path)
        # trainer = pl.Trainer(devices = 1)
        # trainer.test(best_model, test_loader)
        # torch.save(best.state_dict(), "best_model.ckpt")

        best.save_checkpoint(os.path.join("cl_model_" + str(best_acc) + ".ckpt"))
        

        # print(trainer.test(dataloaders=test_loader))

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
        # ['APERIODIC', 'CONSTANT', 'CONTACT_ROT', 'DSCT_BCEP', 'ECLIPSE', 'GDOR_SPB', 'RRLYR_CEPH', 'SOLARLIKE']
        # plt.title(str(best_acc))
        plt.savefig('confusion_matrix_' + str(best_acc) + '.png')
        plt.close()
        # print(len(bModel.testmc))
        # plot misclassifications
        # names = ['APERIODIC', 'CONTACT_ROT', 'CONSTANT', 'DSCT_BCEP', 'ECLIPSE', 'GDOR_SPB', 'RRLYR_CEPH', 'SOLARLIKE']
        names = ['APERIODIC', 'CONSTANT', 'CONTACT_ROT', 'DSCT_BCEP', 'ECLIPSE', 'GDOR_SPB', 'RRLYR_CEPH', 'SOLARLIKE']
        for missede in enumerate(bModel.testmc):
            missed = missede[1]
            # print(missed[0].shape, missed[1].shape)
            data = (missed[1], missed[0])
            # lc = lk.LightCurve({'time':data[0].cpu(), 'flux':data[1].cpu()})
            # power_spec = periodogram.LombScarglePeriodogram.from_lightcurve(lc)
            actual = torch.nonzero(missed[3] == 1).squeeze().item()
            plt.plot(data[0].cpu(), data[1].cpu())
            plt.title('Predicted ' + str({names[prediction[0]]: round(prediction[1], 4) for prediction in enumerate(missed[2].tolist())}) + ', was ' + names[actual])
            plt.xlabel('time')
            plt.ylabel('flux')
            file_path = os.path.join('missclassified_curves', 'missed_curve_lc' + str(missede[0]))
            plt.savefig(file_path)
            plt.close()
            # power_spec.plot()
            # plt.title('power spectrum ' + missede[0] + ': '+ 'predicted ' + names[pred] + ', was ' + names[actual])
            # plt.xlabel('frequency')
            # plt.ylabel('power')
            # file_path = os.path.join('missclassified_curves', 'missed_curve_ps' + str(missede[0]))
            # plt.savefig(file_path)
            # plt.close()
        

if __name__ == '__main__':
     main()