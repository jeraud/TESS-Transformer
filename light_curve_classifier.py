import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

# from wave_scat_conformer import WaveScatFormer
from conformer2 import LCC
# from alternative_transformer import TransformerClassifier

class LightCurveClassifier(pl.LightningModule):
    def __init__(self, transformer_kwargs={"emb":128, "heads":4, "layers":4, "dropout_p":0.05, "hidden":512}, optimizer_kwargs={}, lr=1e-3, num_classes=8, class_weights=None):
        super().__init__()

        self.save_hyperparameters()

        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        print(class_weights.shape)
        print(self.loss_fn)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

        self.classifier = LCC(transformer_kwargs['emb'],transformer_kwargs['heads'],transformer_kwargs['layers'],transformer_kwargs['dropout_p'],transformer_kwargs['hidden'])
        # self.classifier = TransformerClassifier(transformer_kwargs, num_classes=num_classes)

    def forward(self, x, t, mask=None):
        y_pred = self.classifier(x, t, mask)
        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.lr, **self.optimizer_kwargs)
        return {"optimizer": optimizer}
    
    def training_step(self, batch, batch_idx):
        x, t, labels, padding_mask = batch
        y_pred = self(x, t, padding_mask)
        
        loss = self.loss_fn(y_pred, labels)
        #acc = self.accuracy(y_pred, labels)
        
        preds = torch.argmax(y_pred, dim=1)
        acc = self.accuracy(preds, torch.argmax(labels, dim=1))
        
        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True,)
        self.log('train_acc', acc, on_epoch=True, on_step=True, logger=True,sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t, labels, padding_mask = batch
        y_pred = self(x, t, padding_mask)
        
        loss = self.loss_fn(y_pred, labels)
        #acc = self.accuracy(y_pred, labels)
        
        preds = torch.argmax(y_pred, dim=1)
        acc = self.accuracy(preds, torch.argmax(labels, dim=1))
        
        self.log('val_loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True,)
        self.log('val_acc', acc, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    
    def test_step(self, batch, batch_idx):
        x, t, labels, padding_mask = batch
        y_pred = self(x, t, padding_mask)
        
        loss = self.loss_fn(y_pred, labels)
        #acc = self.accuracy(y_pred, labels)
        
        # validation metrics
        preds = torch.argmax(y_pred, dim=1)
        acc = self.accuracy(preds, torch.argmax(labels, dim=1))
        
        self.log('test_loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True,)
        self.log('test_acc', acc, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        return loss