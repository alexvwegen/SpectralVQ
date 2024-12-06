import torch
import torch.nn as nn


class QKVAttention(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.query = nn.Conv1d(channels, channels, kernel_size=1)
        self.key = nn.Conv1d(channels, channels, kernel_size=1)
        self.value = nn.Conv1d(channels, channels, kernel_size=1)
        self.out = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn = torch.softmax(q.transpose(1, 2) @ k / (q.size(1) ** 0.5), dim=-1) # -> (B, T, T)
        out = attn @ v.transpose(1, 2) # -> (B, T, L)
        out = self.out(out.transpose(1, 2))
        return self.dropout(out) + x
    

class FeatureBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.BatchNorm1d(out_channels),
            QKVAttention(out_channels, dropout)
        )

    def forward(self, X):
        return self.block(X)


class ReverseFeatureBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.BatchNorm1d(out_channels),
            QKVAttention(out_channels, dropout)
        )

    def forward(self, X):
        return self.block(X)


class AttentionBlock(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            QKVAttention(channels, dropout),
            nn.BatchNorm1d(channels)
        )

    def forward(self, X):
        return self.block(X)
    

class WeightingLayer(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((num_features, 1)))

    def forward(self, X):
        return torch.einsum("bft, fi->bfit", X, self.weight).squeeze(dim=-2)