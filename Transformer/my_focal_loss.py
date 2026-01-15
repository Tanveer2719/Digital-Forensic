import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for binary/multi-class classification.
    Works with 1D logits for binary classification as well as 2D logits [B, C] for multi-class.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None and (self.smooth < 0 or self.smooth > 1.0):
            raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        # ---- Apply nonlinearity if provided
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)

        # ---- Make logits 2D for binary classification
        if logit.dim() == 1:  # [B] -> [B,1]
            logit = logit.view(-1, 1)

        num_class = logit.shape[1]

        # ---- Flatten extra dims if any (N,C,d1,d2,... -> N*C*m)
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))

        target = target.view(-1, 1)

        # ---- Alpha for class balancing
        alpha = self.alpha
        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1) * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        # ---- One-hot encoding
        idx = target.long()
        one_hot_key = torch.zeros(target.size(0), num_class, device=logit.device)
        one_hot_key.scatter_(1, idx, 1)

        # ---- Smooth labels
        if self.smooth:
            one_hot_key = torch.clamp(one_hot_key, self.smooth/(num_class-1), 1.0 - self.smooth)

        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        alpha = alpha[idx].squeeze()
        loss = -alpha * torch.pow((1 - pt), self.gamma) * logpt

        if self.size_average:
            loss = loss.mean()

        return loss
