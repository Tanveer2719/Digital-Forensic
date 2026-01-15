import torch 
import torch.nn as nn 
class TemporalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(max_len, dim)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        B, T, D = x.shape
        positions = torch.arange(T, device=x.device)
        pos_emb = self.embedding(positions)  # [T, D]
        return x + pos_emb.unsqueeze(0)
