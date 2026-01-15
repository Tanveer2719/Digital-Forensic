import torch
import pytorch_lightning as pl
import torchmetrics
import pandas as pd

from my_AdamW_optimizer import create_optimizer


class ForensicTransMILInterface(pl.LightningModule):

    def __init__(
        self,
        model,
        lr=1e-4,
        weight_decay=1e-4,
        pos_weight=None
    ):
        super().__init__()
        self.model = model
        # self.save_hyperparameters(ignore=["model"])

        # ---- Loss (binary, imbalanced safe)
        if pos_weight is not None:
            self.loss_fn = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weight)
            )
        else:
            self.loss_fn = torch.nn.BCEWithLogitsLoss()

        # ---- Metrics
        self.auroc = torchmetrics.AUROC(task="binary")
        self.f1 = torchmetrics.F1Score(task="binary")
        self.precision = torchmetrics.Precision(task="binary")
        self.recall = torchmetrics.Recall(task="binary")

    # -------------------------------------------------------
    def forward(self, data, mask):
        return self.model(data=data, mask=mask)

    # -------------------------------------------------------
    def training_step(self, batch, batch_idx):
        data, mask, label = batch
        out = self(data, mask)

        logits = out["logits"].squeeze(1)
        label = label.float()

        loss = self.loss_fn(logits, label)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    # -------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        data, mask, label = batch
        out = self(data, mask)

        logits = out["logits"].squeeze(1)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

        label = label.long()

        self.log("val_loss", self.loss_fn(logits, label.float()), prog_bar=True)
        self.log("val_auc", self.auroc(probs, label))
        self.log("val_f1", self.f1(preds, label))
        self.log("val_precision", self.precision(preds, label))
        self.log("val_recall", self.recall(preds, label))

    # -------------------------------------------------------
    def test_step(self, batch, batch_idx):
        data, mask, label = batch
        out = self(data, mask)

        logits = out["logits"].squeeze(1)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

        label = label.long()

        self.log("test_auc", self.auroc(probs, label))
        self.log("test_f1", self.f1(preds, label))
        self.log("test_precision", self.precision(preds, label))
        self.log("test_recall", self.recall(preds, label))

        return {
            "probs": probs.detach(),
            "preds": preds.detach(),
            "labels": label.detach()
        }

    # -------------------------------------------------------
    def configure_optimizers(self):
        return create_optimizer(
            model=self.model,
            lr=self.lr,
            weight_decay=self.weight_decay,
            T_max=20,      # optionally adjust
            eta_min=1e-6
        )
