import sys, os
sys.path.append('./')
import numpy as np
import torch
import pytorch_lightning as pl
import yaml
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from Data.load_qlp import load_qlp_tensors
from Model.light_curve_classifier import LightCurveClassifier
from Helper_functions.train_utils import AccuracyLogger, test_save_best_model, plot_val_acc, make_confusion_matrix, plot_misclassified

def main():
    with open("train_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    pl.seed_everything(42, workers=True)

    # load data
    if config['Data']['tensors_saved_on_device']:
        time, flux, labels = torch.load(config['Data']['time_tensor']), torch.load(config['Data']['flux_tensor']), torch.load(config['Data']['label_tensor'])
    else:
        time, flux, labels = load_qlp_tensors(config['Data']['dir_path'], config['Data']['seq_len'], True, config['Data']['targets'], config['Data']['tess_preprocess'])
    if config['Training']['accelerator'] == 'mps':
        time, flux, labels = time.to(torch.float32), flux.to(torch.float32), labels.to(torch.float32)
    print('data loaded')
    print('time, flux, labels shapes:', time.shape, flux.shape, labels.shape)
    errors = 0
    for label in labels:
        print(label.shape)
        if torch.count_nonzero(label) != 1 or label.shape != torch.Size([7]):
            errors += 1
    print(errors)
    # Get class populations for weighted loss
    total = labels.sum()
    class_weights =  total / labels.sum(axis=0)

    # initialize classifier
    mlce = LightCurveClassifier(transformer_kwargs={'emb':config['Model']['embedding_dim'], 
                                                    'heads':config['Model']['heads'], 
                                                    'layers':config['Model']['layers'],
                                                    'dropout_p':config['Model']['dropout_p'], 
                                                    'hidden':config['Model']['hidden_size'],
                                                    'num_classes':config['Model']['num_classes']}, 
                                optimizer_kwargs= {'lr':config['Training']['learning_rate'],
                                                    'weight_decay':config['Training']['weight_decay']},
                                class_weights=class_weights)

    # creat dataset
    dataset = TensorDataset(flux, time, labels)

    # split data into train, val, test
    nobjects = flux.shape[0]
    n_samples_test = int(config['Training']['test_split'] * nobjects)
    n_samples_val = int(config['Training']['validation_split'] * nobjects)
    train_val_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=n_samples_test,random_state=42, shuffle=True, stratify=labels)
    train_idx, validation_idx = train_test_split(train_val_idx, test_size=n_samples_val,random_state=42, shuffle=True, stratify=labels[train_val_idx])
    dataset_train = Subset(dataset, train_idx)
    dataset_val = Subset(dataset, validation_idx)
    dataset_test = Subset(dataset, test_idx)

    # create dataloaders
    batch_size = config['Training']['batch_size']
    train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=False)

    # initialize callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor=config['Training']['metric'], save_top_k=config['Training']['save_top_k'], mode=config['Training']['mode'], filename='{epoch:02d}-{val_acc:.2f}')
    accuracy_loss_logger = AccuracyLogger()

    print('starting training')
    # initialize trainer
    trainer = trainer = pl.Trainer(max_epochs=config['Training']['max_epochs'],
                            min_epochs=config['Training']['min_epochs'],
                             accelerator=config['Training']['accelerator'],
                             callbacks=[
                                 checkpoint_callback, 
                                 accuracy_loss_logger])
    
    # start training
    trainer.fit(model=mlce, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # test saved model
    best_trainer, best_model, best_test_acc, best_test_loss = test_save_best_model(checkpoint_callback.best_k_models, test_loader)
    print('Test Accuracy:', best_test_acc)
    print('Test Loss:', best_test_loss)
    rounded_test_acc = round(best_test_acc, 4)

    # save best model
    best_trainer.save_checkpoint(os.path.join(config['Training']['output_dir'], 'LCC_test_acc_' + str(rounded_test_acc) + '.ckpt'))

    # plot train, validation accuracy
    print('plotting val accuracy')
    plot_val_acc(accuracy_loss_logger, config['Training']['output_dir'])

    # make confusion matrix
    print('making confusion matrix')
    make_confusion_matrix(best_model, config['Model']['class_names'], config['Training']['accelerator'], config['Training']['output_dir'], rounded_test_acc)
    if config['Training']['plot_misclassified']:
        # plot misclassified light curves
        print('plotting misclassified curves') 
        plot_misclassified(best_model, config['Training']['output_dir'], config['Model']['class_names'])
    print('finished')
if __name__ == "__main__":
    main()













