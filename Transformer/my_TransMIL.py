import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention
from my_Temporal_Poistional_Encoding import TemporalPositionalEncoding


class TransLayer(nn.Module):
    """Single Transformer Layer with NystromAttention."""

    def __init__(self, dim=512, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1,
        )

    def forward(self, x):
        return x + self.attn(self.norm(x))


class TransMIL(nn.Module):
    """Transformer-based Multiple Instance Learning (TransMIL)."""

    def __init__(self, input_dim, dim=512, n_layers=2):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU()
        )

        self.pos_encoding = TemporalPositionalEncoding(dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.layers = nn.ModuleList([TransLayer(dim=dim) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.fc_out = nn.Linear(dim, 1)
        print('Inside the __main__function')

    def forward(self, data, mask=None):
        """
        Args:
            data : Tensor[B, T, D] - input bag of instances
            mask : Tensor[B, T] or None - optional mask for variable-length sequences

        Returns:
            dict with keys:
                logits : [B] raw outputs
                Y_prob : [B] probabilities
                Y_hat  : [B] predicted labels
        """
        B, T, _ = data.shape

        # ---- Initial embedding
        h = self.fc1(data)
        h = self.pos_encoding(h)

        # ---- Apply mask if given
        if mask is not None:
            h = h * mask.unsqueeze(-1)

        # ---- Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)

        # ---- Transformer layers
        for layer in self.layers:
            h = layer(h)
            if mask is not None:
                h[:, 1:] = h[:, 1:] * mask.unsqueeze(-1)

        # ---- LayerNorm
        h = self.norm(h)
        cls_out = h[:, 0]  # CLS token output

        # ---- Final classification
        logits = self.fc_out(cls_out)

        # ---- Squeeze safely
        if logits.dim() == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)

        Y_prob = torch.sigmoid(logits)
        Y_hat = (Y_prob > 0.5).long()

        return {
            "logits": logits,
            "Y_prob": Y_prob,
            "Y_hat": Y_hat
        }
