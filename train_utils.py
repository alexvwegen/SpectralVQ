import torch
import torch.nn.functional as F


def embedding_loss(encoder_out, codebook_out):
    return torch.mean((encoder_out.detach() - codebook_out) ** 2)

def commitment_loss(encoder_out, codebook_out):
    return torch.mean((encoder_out - codebook_out.detach()) ** 2)

def vq_loss(embed_loss, commit_loss, beta):
    return embed_loss + beta * commit_loss

def reconstruction_loss(x, y):
    return F.mse_loss(y, x)