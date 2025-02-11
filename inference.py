import os
import yaml
import torch
import pytorch_lightning as pl
import lightkurve as lk
from Helper_functions.create_plots import create_plot
from Data.load_qlp import load_qlp_tensors, load_qlp_curve_from_file
from torch.utils.data import DataLoader, TensorDataset, Subset
from Model.light_curve_classifier import LightCurveClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def main():
    with open("inference_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    if config['Data']['tensors_saved_on_device']:
        time, flux, ticids = torch.load(config['Data']['time_tensor']), torch.load(config['Data']['flux_tensor']), torch.load(config['Data']['tic_tensor'])
    else:
        time, flux, ticids = load_qlp_tensors(config['Data']['data_path'], config['Data']['seq_len'], need_preprocess=config['Data']['tess_preprocess'])

    if config['Inference']['device'] == 'mps':
        time, flux, ticids = time.to(torch.float32), flux.to(torch.float32), ticids.to(torch.float32)
    
    dataset = TensorDataset(flux, time, ticids)
    data_loader = DataLoader(dataset, batch_size=config['Inference']['batch_size'], num_workers=9, pin_memory=True, shuffle=False)

    model = LightCurveClassifier.load_from_checkpoint(config['Model']['weights_path'])
    if config['Inference']['device'] == 'mps':
        model = model.to(torch.float32)

    os.makedirs(config['Inference']['output_dir'],exist_ok=True)
    trainer = pl.Trainer()
    trainer.predict(model, data_loader)
    names = config['Model']['class_names']
    produce_plots = config['Inference']['produce_plots']
    output_dir = config['Inference']['output_dir']
    if produce_plots:
        for name in names:
            plot_path_dir = os.path.join(output_dir, name + '/')
            os.makedirs(os.path.dirname(plot_path_dir), exist_ok=True)
    with open(output_dir + 'predictions.txt', 'a') as f:
        for prediction in model.predicted:
            tic, flux, time, softmax = prediction
            probabilities = str({names[idx]: round(prob, 4) for idx, prob in enumerate(softmax.tolist())})
            f.write(str(tic.item()) + ' ' + probabilities + '\n')
            if produce_plots:
                most_likely = names[torch.argmax(softmax)]
                lc = lk.LightCurve(time = time.cpu(), flux = flux.cpu())
                title = 'Predicted: ' + probabilities
                plot_path = os.path.join(output_dir, most_likely + '/', str(int(tic.item()))+'_plot.png')
                create_plot(lc, plot_path,  title)




if __name__ == '__main__':
    main()
