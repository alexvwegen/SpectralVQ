import torch
import torch.nn as nn

from blocks import *


class Encoder(nn.Module):
    def __init__(self, input_dim, num_bins, latent_dim=512, dropout=0.1):
        super().__init__()
        self.projection = WeightingLayer(input_dim)
        self.feature_blocks = nn.Sequential(
            *[
                FeatureBlock(num_bins, latent_dim // 4, dropout=dropout),
                FeatureBlock(latent_dim // 4, latent_dim // 2, dropout=dropout),
                FeatureBlock(latent_dim // 2, latent_dim, dropout=dropout)
            ]
        )

        self.residual_blocks = nn.Sequential(
            *[AttentionBlock(latent_dim, dropout) for _ in range(2)]
        )

    def forward(self, X):
        z = self.projection(X) # -> (B, F, T)
        z = self.feature_blocks(z) # -> (B, F, TL)
        z = self.residual_blocks(z) # -> (B, L, TL)
        return z


class Codebook(nn.Module):
    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings

    def tokenize(self, x):
        x = x.transpose(1, 2) # -> (B, TL, L)
        x_flat = x.reshape(-1, x.size(-1)) # -> (B * TL, L)
        distances = ((x_flat.unsqueeze(1) - self.embedding.weight) ** 2).sum(-1)
        indices = distances.argmin(1)
        return indices.view(*x.size()[:-1])  # Reshape back to input dims

    def forward(self, x):
        indices = self.tokenize(x)
        quantized = self.embedding(indices).transpose(1, 2)
        return quantized, indices


class Decoder(nn.Module):
    def __init__(self, num_bins, latent_dim=512, dropout=0.1):
        super().__init__()

        self.feature_blocks = nn.Sequential(
            *[
                ReverseFeatureBlock(latent_dim, latent_dim // 2, dropout=dropout),
                ReverseFeatureBlock(latent_dim // 2, latent_dim // 4, dropout=dropout),
                ReverseFeatureBlock(latent_dim // 4, num_bins, dropout=dropout)
            ]
        )

        self.residual_blocks = nn.Sequential(
            *[AttentionBlock(latent_dim, dropout) for _ in range(2)]
        )

        self.refinement = nn.Linear(num_bins, num_bins)

    def forward(self, x):
        x = self.residual_blocks(x) # -> (B, L, TL)
        x = self.feature_blocks(x) # -> (B, F, T)
        x = self.refinement(x.transpose(1, 2)) # -> (B, T, F)
        return x.transpose(1, 2) # -> (B, F, T)


class VQVAE(nn.Module):
    def __init__(self, num_bins, num_embeddings, latent_dim=512, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(input_dim=num_bins, num_bins=num_bins, latent_dim=latent_dim, dropout=dropout)
        self.codebook = Codebook(embedding_dim=latent_dim, num_embeddings=num_embeddings)
        self.decoder = Decoder(num_bins=num_bins, latent_dim=latent_dim, dropout=dropout)

    def forward(self, x):
        z = self.encoder(x)
        z_quantized, indices = self.codebook(z)
        recon = self.decoder(z_quantized)
        return recon, z, z_quantized, indices