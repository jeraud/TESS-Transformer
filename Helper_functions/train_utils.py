import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
import os
import torch
from Model.light_curve_classifier import LightCurveClassifier
import lightkurve as lk
from Helper_functions.create_plots import create_plot

class AccuracyLogger(Callback):
    def __init__(self):
        self.val_acc = []

    def on_validation_end(self, trainer, pl_module):
        self.val_acc.append(trainer.callback_metrics['val_acc'].item())


def test_save_best_model(top_k_checkpoints, test_loader):
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
    return best, bModel, best_acc, best_loss

def plot_val_acc(accuracy_logger, output_dir):
    # plot validation accuracy
    epochs = range(1, len(accuracy_logger.val_acc) + 1)
    plt.plot(epochs, accuracy_logger.val_acc, 'b', label='Validation accuracy')
    plt.title('Validation Accuracy During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'val_acc.png'))    
    plt.close()

def make_confusion_matrix(model, class_names, device, outputdir, rounded_acc):
    metric = model.conf_matrix.to(device)
    fig_, ax_ = metric.plot(labels=class_names)
        # ['APERIODIC', 'CONSTANT', 'CONTACT_ROT', 'DSCT_BCEP', 'ECLIPSE', 'GDOR_SPB', 'RRLYR_CEPH', 'SOLARLIKE']
        # plt.title(str(best_acc))
    plt.savefig(os.path.join(outputdir, 'confusion_matrix_' + str(rounded_acc) + '.png'))
    plt.close()


def plot_misclassified(model, outputdir, class_names):
    misclassified_folder = os.makedirs(os.path.join(outputdir, 'misclassified_curves/'), exist_ok=True)
    for missede in enumerate(model.testmc):
            missed = missede[1]
            # print(missed[0].shape, missed[1].shape)
            data = (missed[1], missed[0])
            # lc = lk.LightCurve({'time':data[0].cpu(), 'flux':data[1].cpu()})
            # power_spec = periodogram.LombScarglePeriodogram.from_lightcurve(lc)
            actual = torch.nonzero(missed[3] == 1).squeeze().item()
            title = 'Predicted ' + str({class_names[prediction[0]]: round(prediction[1], 4) for prediction in enumerate(missed[2].tolist())}) + ', was ' + class_names[actual]
            lc = lk.LightCurve(time = data[0].cpu(), flux = data[1].cpu())
            file_path = os.path.join(os.path.join(outputdir, 'misclassified_curves/'), 'missed_curve_lc' + str(missede[0]))
            create_plot(lc, file_path, title)
            # plt.plot(data[0].cpu(), data[1].cpu())
            # plt.title('Predicted ' + str({class_names[prediction[0]]: round(prediction[1], 4) for prediction in enumerate(missed[2].tolist())}) + ', was ' + class_names[actual])
            # plt.xlabel('time')
            # plt.ylabel('flux')
            # file_path = os.path.join(os.path.join(outputdir, 'misclassified_curves/'), 'missed_curve_lc' + str(missede[0]))
            # plt.savefig(file_path)
            # plt.close()
            