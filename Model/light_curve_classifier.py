import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LambdaLR
from Model.lsatt_conv import LCC

class LightCurveClassifier(pl.LightningModule):
    def __init__(self, transformer_kwargs={"emb":64, "heads":8, "layers":3, "dropout_p":0.1, "hidden":512, "num_classes":8}, optimizer_kwargs={"lr":1e-4, "weight_decay":1e-5},  class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.lr = optimizer_kwargs['lr']
        self.weight_decay = optimizer_kwargs['weight_decay']
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=transformer_kwargs['num_classes'],
        )
        self.test_preds = []
        self.test_labels = []
        self.conf_matrix = MulticlassConfusionMatrix(num_classes=transformer_kwargs['num_classes'])
        self.classifier = LCC(emb_d=transformer_kwargs['emb'], num_heads=transformer_kwargs['heads'], layers=transformer_kwargs['layers'], dropout_p=transformer_kwargs['dropout_p'], ffn_d=transformer_kwargs['hidden'], num_classes=transformer_kwargs['num_classes'])
        self.testmc = []
        self.predicted = []
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, t, mask=None):
        y_pred = self.classifier(x, t, mask)
        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, 
                                      weight_decay=self.weight_decay)
        return [optimizer]
    
    def training_step(self, batch, batch_idx):
        x, t, labels = batch
        y_pred = self.forward(x, t)
        
        loss = self.loss_fn(y_pred, labels)        
        preds = torch.argmax(y_pred, dim=1)
        acc = self.accuracy(preds, torch.argmax(labels, dim=1))

        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log('train_acc', acc, on_epoch=True, on_step=True, logger=True,sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t, labels = batch
        y_pred = self.forward(x, t)
        
        loss = self.loss_fn(y_pred, labels)
        
        preds = torch.argmax(y_pred, dim=1)
        acc = self.accuracy(preds, torch.argmax(labels, dim=1))
        
        self.log('val_loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    
    def test_step(self, batch, batch_idx):
        x, t, labels = batch
        y_pred = self.forward(x, t)
        loss = self.loss_fn(y_pred, labels)
        # validation metrics
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
                self.testmc.append((x[i], t[i], y_pred[i], labels[i]))
        self.log('test_loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log('test_acc', acc, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        return loss
    
    def predict_step(self, batch):
        x, t, tic = batch
        y_pred_softmax = self.forward(x,t)
        for i in range(len(y_pred_softmax)):
            self.predicted.append((tic[i], x[i], t[i], y_pred_softmax[i]))
        return y_pred_softmax

