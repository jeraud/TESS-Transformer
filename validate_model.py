import numpy as np
import torch
import lightkurve as lk
import os
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
from light_curve_classifier import LightCurveClassifier
import pytorch_lightning as pl
from create_plots import create_plot
from sklearn.model_selection import train_test_split
import torch

def validate():
    # load validation set flux, time labels
    flux = torch.load('fluxtensor3.pt')
    time = torch.load('timetensor3.pt')
    labels = torch.load('labelstensor3.pt')
    # need to convert to float 32 to run on mps istead of cuda
    flux = torch.tensor(flux, dtype=torch.float32)
    time = torch.tensor(time, dtype=torch.float32)
    dataset = TensorDataset(flux, time, labels)
    nobjects = flux.shape[0]
    val_fraction = 0.1
    test_fraction = 0.05
    batch_size = 32
    n_samples_test = int(test_fraction * nobjects)
    n_samples_val = int(val_fraction * nobjects)

    train_val_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=n_samples_test,random_state=42, shuffle=True, stratify=labels)
    train_idx, validation_idx = train_test_split(train_val_idx, test_size=n_samples_val,random_state=42, shuffle=True, stratify=labels[train_val_idx])

    dataset_train = Subset(dataset, train_idx)
    dataset_val = Subset(dataset, validation_idx)
    dataset_test = Subset(dataset, test_idx)
    # load model to test
    model = LightCurveClassifier.load_from_checkpoint('cl_model_0.884.ckpt')
    test_loader = DataLoader(dataset, batch_size=32, num_workers=9, pin_memory=True, shuffle=False)
    trainer = pl.Trainer()
    trainer.test(model, test_loader)
    names = ['APERIODIC', 'CONSTANT', 'CONTACT_ROT', 'DSCT_BCEP', 'ECLIPSE', 'GDOR_SPB', 'RRLYR_CEPH', 'SOLARLIKE']
    print(len(model.testmc))
    # plot light curves, power spectrum for missclassified curves
    for missede in enumerate(model.testmc):
        missed = missede[1]
        data1 = (np.array(missed[1].cpu()), np.array(missed[0].cpu()))
        lc = lk.LightCurve(time = data1[0], flux = data1[1])
        pred = missed[2]
        probs = {names[i]:round(pred[i].item(),3) for i in range(8)}
        actual = torch.nonzero(missed[3] == 1).squeeze().item()
        title = 'Actual class: ' + names[actual] + ', output probabilities: ' + str(probs)
        path = os.path.join(names[actual], 'missed_curve' + str(missede[0]))
        create_plot(lc, path,  title)
if __name__ == '__main__':
    validate()