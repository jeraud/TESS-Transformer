import numpy as np
import torch
import lightkurve as lk
import os
from torch.utils.data import TensorDataset, DataLoader
from light_curve_classifier import LightCurveClassifier
import pytorch_lightning as pl
from create_plots import create_plot
import torch


def classify(data, model, folder_path):
    """
    predict class of a set of lightcurves: run model on data, store results, plots in folder_path

    @args:
        data(tuple[string]): tuple of (time, flux) where each is a path to time, flux tensors (.pt files)

        model(string): The path to a .ckpt model checkpoint file you wish to load the weights from
        folder_path(string): The path to a folder you want to store plots with light curve, 
                            power spectrum plots, predicted outputs for each light curve you want to predict.
    """
    time, flux = torch.load(data[0]), torch.load(data[1])

    # if no cuda, need float32 to run on MPS
    if not torch.cuda.is_available():
        flux = torch.tensor(flux, dtype=torch.float32)
        time = torch.tensor(time, dtype=torch.float32)
    
    # create dataset
    dataset = TensorDataset(flux, time)

    # load model from checkpoint
    model = LightCurveClassifier.load_from_checkpoint(model)

    # load data
    data_loader = DataLoader(dataset, batch_size=32, num_workers=9, pin_memory=True, shuffle=False)

    # initialize lightning trainer, predict with trainer
    trainer = pl.Trainer()
    trainer.predict(model, data_loader)
    

    # plot light curves, power spectrum for predicted curves
    names = ['APERIODIC', 'CONSTANT', 'CONTACT_ROT', 'DSCT_BCEP', 'ECLIPSE', 'GDOR_SPB', 'RRLYR_CEPH', 'SOLARLIKE']
    for predicted_enum in enumerate(model.predicted):
        light_curve_prediction = predicted_enum[1]
        data1 = (np.array(light_curve_prediction[1].cpu()), np.array(light_curve_prediction[0].cpu()))
        lc = lk.LightCurve(time = data1[0], flux = data1[1])
        pred = light_curve_prediction[2]
        softmax = light_curve_prediction[3]
        probs = {names[i]:round(softmax[i].item(),3) for i in range(8)}
        title = 'Predicted: ' + names[pred] + '; softmax output: ' + str(probs) 
        path = os.path.join(folder_path, str(predicted_enum[0]))
        create_plot(lc, path,  title)

if __name__ == '__main__':
    classify(('combotimetensor.pt', 'combofluxtensor.pt'), 'cl_model_0.884.ckpt', '/Users/paul/Desktop/UROP/tess-wavelets/Predictions')

