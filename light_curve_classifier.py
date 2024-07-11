import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LambdaLR

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

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_preds = []
        self.test_labels = []
        self.conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
        self.classifier = LCC(transformer_kwargs['emb'],transformer_kwargs['heads'],transformer_kwargs['layers'],transformer_kwargs['dropout_p'],transformer_kwargs['hidden'])
        self.testmc = []
    def forward(self, x, t, mask=None):
        y_pred = self.classifier(x, t, mask)
        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.lr, weight_decay=1e-5, **self.optimizer_kwargs)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        return {"optimizer": optimizer, "scheduler": scheduler}
    
    def training_step(self, batch, batch_idx):
        x, t, labels, padding_mask = batch
        y_pred = self.forward(x, t, padding_mask)
        
        loss = self.loss_fn(y_pred, labels)
        
        preds = torch.argmax(y_pred, dim=1)
        acc = self.accuracy(preds, torch.argmax(labels, dim=1))
        
        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True,)
        self.log('train_acc', acc, on_epoch=True, on_step=True, logger=True,sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t, labels, padding_mask = batch
        y_pred = self.forward(x, t, padding_mask)
        
        loss = self.loss_fn(y_pred, labels)
        
        preds = torch.argmax(y_pred, dim=1)
        acc = self.accuracy(preds, torch.argmax(labels, dim=1))
        
        self.log('val_loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True,)
        self.log('val_acc', acc, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    
    def test_step(self, batch, batch_idx):
        x, t, labels, padding_mask = batch
        y_pred = self.forward(x, t, padding_mask)
        loss = self.loss_fn(y_pred, labels)
        preds = torch.argmax(y_pred, dim=1) 
        pred2 = torch.argmax(y_pred, dim=1, keepdim=True)
        labels2 = labels.argmax(dim=1)
        self.conf_matrix.update(preds.to(self.conf_matrix.device),labels2.to(self.conf_matrix.device))
        acc = self.accuracy(preds, torch.argmax(labels, dim=1))
        self.test_preds = pred2
        self.test_labels = labels
        def misclassified(pred,y):
            return y[pred.item()] != 1
        for i in range(len(pred2)):
            if misclassified(pred2[i], labels[i]):
                self.testmc.append((x[i], t[i], pred2[i], labels[i]))
        self.log('test_loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True,)
        self.log('test_acc', acc, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        return loss
    