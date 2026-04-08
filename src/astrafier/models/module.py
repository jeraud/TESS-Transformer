"""Lightning module that stitches together the encoder and classification head."""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix

from .head import ClassificationHead
from .light_curve_classifier import LightCurveEncoder


class AstrafierModule(pl.LightningModule):
    """Full model used for training and inference.

    The module couples the transformer-based encoder with the feed-forward head
    and exposes the training, validation, test, and prediction steps used in
    the original research code. Behaviour is kept identical to the prior
    implementation – only the structure and documentation have been improved.
    """

    def __init__(
        self,
        class_weight: Optional[torch.Tensor] = None,
        layers: int = 5,
        mc_dropout: bool = True,
        mc_samples: int = 20,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weight"])
        self.layers = layers
        transformer_kwargs = {
            "emb": 64,
            "heads": 8,
            "layers": 3,
            "dropout_p": 0.2,
            "hidden": 256,
            "num_classes": 8,
        }
        self.model = LightCurveEncoder(transformer_kwargs)
        self.head = ClassificationHead()
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weight)
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=8)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=8)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=8)
        self.conf_matrix = MulticlassConfusionMatrix(num_classes=8)
        self.umap_embs: list[torch.Tensor] = []
        self.umap_labels: list[torch.Tensor] = []
        self.testmc: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.mc_dropout_enabled = mc_dropout
        self.mc_samples = max(1, mc_samples)
        self._mc_dropout_rate = 0.2
        self._mc_dropout_layers: list[nn.Dropout] = []
        self._mc_original_p: list[float] = []
        self._collect_dropout_layers()
        self._warmup_epochs = 5
        self._warmup_steps = self._warmup_epochs * 100

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self.model(x, t, mask)
        return self.head(z)

    # Lightning API -----------------------------------------------------------------

    def configure_optimizers(self):  # type: ignore[override]
        opt = torch.optim.AdamW(
            [
                {"params": self.model.parameters(), "lr": 1e-4},
                {"params": self.head.parameters(), "lr": 1e-3},
            ],
            weight_decay=1e-5,
            betas=(0.9, 0.95),
        )

        def lr_lambda(step: int) -> float:
            warmup_steps = max(1, self._warmup_steps)
            return min(1.0, (step + 1) / warmup_steps)

        warmup = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=5, verbose=True, min_lr=1e-6
        )

        scheds: Iterable[dict[str, object]] = [
            {"scheduler": warmup, "interval": "step"},
            {"scheduler": plateau, "monitor": "val_loss"},
        ]
        return [opt], list(scheds)

    def on_fit_start(self) -> None:  # type: ignore[override]
        self._warmup_steps = self._estimate_warmup_steps()

    def _estimate_warmup_steps(self) -> int:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return self._warmup_steps
        batches_per_epoch = getattr(trainer, "num_training_batches", None)
        if isinstance(batches_per_epoch, int) and batches_per_epoch > 0:
            return max(1, batches_per_epoch * self._warmup_epochs)
        estimated_batches = getattr(trainer, "estimated_stepping_batches", None)
        max_epochs = getattr(trainer, "max_epochs", None)
        if (
            isinstance(estimated_batches, int)
            and estimated_batches > 0
            and isinstance(max_epochs, int)
            and max_epochs > 0
        ):
            warmup_epochs = min(self._warmup_epochs, max_epochs)
            warmup_ratio = warmup_epochs / max_epochs
            return max(1, int(estimated_batches * warmup_ratio))
        return self._warmup_steps

    def training_step(self, batch, batch_idx):  # type: ignore[override]
        x, t, labels, mask = batch
        logits = self.forward(x, t, mask)
        y = labels.argmax(dim=1).long()
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        acc = self.train_acc(preds, y)
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log("train_acc", acc, on_epoch=True, on_step=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore[override]
        x, t, labels, mask = batch
        logits = self.forward(x, t, mask)
        y = labels.argmax(dim=1).long()
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        acc = self.val_acc(preds, y)
        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):  # type: ignore[override]
        x, t, labels, mask = batch
        logits = self.forward(x, t, mask)
        y = labels.argmax(dim=1).long()
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        acc = self.test_acc(preds, y)
        self.conf_matrix.update(preds, y)
        for i in range(len(preds)):
            if preds[i] != y[i]:
                self.testmc.append((x[i], t[i], logits[i], labels[i]))
        self.log("test_loss_epoch", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_acc_epoch", acc, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    # Predict API -------------------------------------------------------------------

    def _collect_dropout_layers(self) -> None:
        self._mc_dropout_layers.clear()
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                self._mc_dropout_layers.append(module)
        for module in self.head.modules():
            if isinstance(module, nn.Dropout):
                self._mc_dropout_layers.append(module)

    def _set_mc_dropout_state(self, enabled: bool) -> None:
        if enabled:
            self._mc_original_p = [layer.p for layer in self._mc_dropout_layers]
            for layer in self._mc_dropout_layers:
                layer.p = self._mc_dropout_rate
                layer.train()
        else:
            for layer, original_p in zip(self._mc_dropout_layers, self._mc_original_p):
                layer.p = original_p
                layer.eval()
            self._mc_original_p = []

    def on_predict_start(self) -> None:  # type: ignore[override]
        if self.mc_dropout_enabled:
            self._collect_dropout_layers()
            self._set_mc_dropout_state(True)
        else:
            self.eval()

    def on_predict_end(self) -> None:  # type: ignore[override]
        if self.mc_dropout_enabled:
            self._set_mc_dropout_state(False)
        self.model.eval()
        self.head.eval()

    def on_test_epoch_start(self) -> None:  # type: ignore[override]
        self.testmc.clear()

    def predict_step(self, batch, batch_idx=None):  # type: ignore[override]
        x, t, tic, mask = batch
        samples = self.mc_samples if self.mc_dropout_enabled else 1
        with torch.no_grad():
            if samples > 1:
                preds = [
                    F.softmax(self.head(self.model(x, t, mask)), dim=-1) for _ in range(samples)
                ]
                monte_carlo_dropout_preds = torch.stack(preds)
                p_mean = monte_carlo_dropout_preds.mean(dim=0)
                p_var = monte_carlo_dropout_preds.var(dim=0)
                s_hat = p_mean * (1.0 - p_mean) / (p_var + 1e-12) - 1.0
                S = s_hat.clamp_min(0.0).mean(dim=-1, keepdim=True)
                alpha = 1.0 + S * p_mean
                p_dirichlet = alpha / alpha.sum(dim=-1, keepdim=True)
            else:
                logits = self.head(self.model(x, t, mask))
                p_dirichlet = F.softmax(logits, dim=-1)
                p_mean = p_dirichlet

        batch_outputs: List[Dict[str, object]] = []
        for i in range(len(p_mean)):
            batch_outputs.append(
                {
                    "tic": int(tic[i].item()),
                    "probabilities": p_dirichlet[i].detach().cpu().tolist(),
                }
            )
        return batch_outputs

