"""Training utilities shared between scripts."""

from __future__ import annotations

import os
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from astrafier.models import AstrafierModule


class AccuracyLogger(Callback):
    """Callback that records validation accuracy at the end of each epoch."""

    def __init__(self) -> None:
        super().__init__()
        self.val_acc: list[float] = []

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metric = trainer.callback_metrics.get("val_acc")
        if metric is not None:
            self.val_acc.append(metric.item())


def test_save_best_model(top_k_checkpoints: Iterable[str], test_loader: torch.utils.data.DataLoader, class_weights: torch.Tensor | None = None):
    best_acc = 0.0
    best_loss = 0.0
    best_trainer: pl.Trainer | None = None
    best_model: AstrafierModule | None = None

    for checkpoint_path in top_k_checkpoints:
        model = AstrafierModule.load_from_checkpoint(checkpoint_path, strict=False, class_weight=class_weights)
        trainer = pl.Trainer()
        test = trainer.test(model, test_loader)
        test_acc = test[0]["test_acc_epoch"]
        test_loss = test[0]["test_loss_epoch"]
        if (test_acc > best_acc) or (test_acc == best_acc and test_loss < best_loss):
            best_acc = test_acc
            best_loss = test_loss
            best_trainer = trainer
            best_model = model
    return best_trainer, best_model, best_acc, best_loss


def plot_val_acc(accuracy_logger: AccuracyLogger, output_dir: str) -> None:
    epochs = range(1, len(accuracy_logger.val_acc) + 1)
    plt.plot(epochs, accuracy_logger.val_acc, "b", label="Validation accuracy")
    plt.title("Validation Accuracy During Training")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "val_acc.png"))
    plt.close()


def make_confusion_matrix(model: AstrafierModule, class_names: Sequence[str], device: torch.device | str, output_dir: str, rounded_acc: float) -> None:
    metric = model.conf_matrix.to(device)
    fig_, ax_ = metric.plot(labels=list(class_names))
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{rounded_acc}.png"))
    plt.close()


def plot_misclassified(model: AstrafierModule, output_dir: str, class_names: Sequence[str]) -> None:
    import lightkurve as lk  # local import to avoid heavy dependency at module import time

    os.makedirs(os.path.join(output_dir, "misclassified_curves"), exist_ok=True)
    for idx, missed in enumerate(model.testmc):
        data = (missed[1], missed[0])
        actual = torch.nonzero(missed[3] == 1).squeeze().item()
        title = "Predicted " + str({class_names[prediction[0]]: round(prediction[1], 4) for prediction in enumerate(missed[2].tolist())}) + ", was " + class_names[actual]
        lc = lk.LightCurve(time=data[0].cpu(), flux=data[1].cpu())
        file_path = os.path.join(output_dir, "misclassified_curves", f"missed_curve_lc{idx}")
        from astrafier.utils.plots import create_plot

        create_plot(lc, file_path, title)

