import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
from nystrom_attention import NystromAttention
from my_Temporal_Poistional_Encoding import TemporalPositionalEncoding


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x

class TransMIL(nn.Module):
    def __init__(self, input_dim, dim=512, n_layers=2):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU()
        )

        self.pos_encoding = TemporalPositionalEncoding(dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.layers = nn.ModuleList([
            TransLayer(dim=dim) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(dim)
        self.fc_out = nn.Linear(dim, 1)



    def forward(self, data, mask=None):
        B, T, _ = data.shape

        h = self.fc1(data)
        h = self.pos_encoding(h)

        if mask is not None:
            h = h * mask.unsqueeze(-1)

        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)

        for layer in self.layers:
            h = layer(h)
            if mask is not None:
                h[:, 1:] = h[:, 1:] * mask.unsqueeze(-1)

        h = self.norm(h)
        cls_out = h[:, 0]

        logits = self.fc_out(cls_out).squeeze(1)

        Y_prob = torch.sigmoid(logits)
        Y_hat = (Y_prob > 0.5).long()

        return {
            "logits": logits,
            "Y_prob": Y_prob,
            "Y_hat": Y_hat
        }
