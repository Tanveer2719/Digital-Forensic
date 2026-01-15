import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention
from my_Temporal_Poistional_Encoding import TemporalPositionalEncoding


class TransLayer(nn.Module):
    """Single Transformer Layer with NystromAttention and Dropout."""

    def __init__(self, dim=512, norm_layer=nn.LayerNorm, dropout=0.1):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        x = self.dropout(x)
        return x


class TransMIL(nn.Module):
    """Transformer-based Multiple Instance Learning (TransMIL) with CLS + Mean Pooling and Dropout."""

    def __init__(self, input_dim, dim=512, n_layers=2, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout)  # input embedding dropout
        )

        self.pos_encoding = TemporalPositionalEncoding(dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.layers = nn.ModuleList([TransLayer(dim=dim, dropout=dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)

        # Use both CLS token + mean pooling for stability
        self.fc_out = nn.Linear(dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data, mask=None):
        B, T, _ = data.shape

        # ---- Input embedding
        h = self.fc1(data)
        h = self.pos_encoding(h)

        # ---- Apply mask if provided
        if mask is not None:
            h = h * mask.unsqueeze(-1)

        # ---- CLS token
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)

        # ---- Transformer layers
        for layer in self.layers:
            h = layer(h)
            if mask is not None:
                h[:, 1:] = h[:, 1:] * mask.unsqueeze(-1)

        h = self.norm(h)
        cls_out = h[:, 0]
        mean_out = h[:, 1:].mean(dim=1)
        pooled_out = cls_out + mean_out  # combine CLS + mean pooling

        pooled_out = self.dropout(pooled_out)

        # ---- Output
        logits = self.fc_out(pooled_out)
        if logits.dim() == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)

        Y_prob = torch.sigmoid(logits)
        Y_hat = (Y_prob > 0.5).long()

        return {"logits": logits, "Y_prob": Y_prob, "Y_hat": Y_hat}
