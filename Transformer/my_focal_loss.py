import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Binary Focal Loss operating on probabilities (after sigmoid).

    Expected:
        probs  : Tensor (B,) or (B,1) in [0,1]
        targets: Tensor (B,) or (B,1) in {0,1}
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean", eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, probs, targets):
        probs = probs.view(-1)
        targets = targets.view(-1).float()

        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)

        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        loss = -alpha_t * (1 - pt) ** self.gamma * torch.log(pt)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
