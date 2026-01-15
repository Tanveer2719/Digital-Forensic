import torch
import pytorch_lightning as pl
import torchmetrics
from my_AdamW_optimizer import create_optimizer
from my_focal_loss import FocalLoss  

class ForensicTransMILInterface(pl.LightningModule):
    def __init__(self, model, lr=5e-5, weight_decay=1e-4,
                 use_focal=False, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.model = model

        # Save hyperparameters manually
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_focal = use_focal

        # ---- Loss
        if use_focal:
            # Use Focal Loss for imbalanced MIL datasets
            self.loss_fn = FocalLoss(apply_nonlin=torch.sigmoid, alpha=alpha, gamma=gamma)
        else:
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
    def forward(self, data, mask=None):
        return self.model(data=data, mask=mask)

    # -------------------------------------------------------
    def compute_loss_and_preds(self, batch):
        data, mask, label = batch
        out = self(data, mask)

        logits = out["logits"]
        if logits.dim() == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        return logits, probs, preds, label

    # -------------------------------------------------------
    def training_step(self, batch, batch_idx):
        logits, probs, preds, label = self.compute_loss_and_preds(batch)
        loss = self.loss_fn(logits, label.float())
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    # -------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        logits, probs, preds, label = self.compute_loss_and_preds(batch)

        label = label.long()
        self.log("val_loss", self.loss_fn(logits, label.float()), prog_bar=True)
        self.log("val_auc", self.auroc(probs, label))
        self.log("val_f1", self.f1(preds, label))
        self.log("val_precision", self.precision(preds, label))
        self.log("val_recall", self.recall(preds, label))

    # -------------------------------------------------------
    def test_step(self, batch, batch_idx):
        logits, probs, preds, label = self.compute_loss_and_preds(batch)

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
            T_max=20,
            eta_min=1e-6
        )
