"""
Implements the three loss functions.
For reconstruction loss:
    input: batch_size*B, spectra of batches of (center) pixels
    pred: batch_size*B, the reconstructed spectra

KL Divergence loss: concatenated mean and log var

Homology loss: x_ls and x_ss

"""
import torch
import torch.nn as nn

def reconstruction_loss(input_vec: torch.Tensor, pred_vec: torch.Tensor, criterion:str = "mse" ) ->torch.Tensor:
    if criterion == "mse":
        loss = nn.MSELoss()
    elif criterion == "huber":
        loss = nn.HuberLoss()
    else:
        raise ValueError("Valid reconstruction criterion are 'mse' for MSE loss or 'huber' for Huber Loss.")
    return loss(input_vec, pred_vec)

def kl_loss(revised_mean: torch.Tensor, log_var: torch.Tensor, free_bits: float =0.0) -> torch.Tensor:
    # compute per-sample, per-dim kl, with free-bits
    kl_pd = 0.5*(torch.exp(log_var) + torch.square(revised_mean) - 1 - log_var)
    # clamp each dimension to at least free_bits
    kl_fb = torch.clamp(kl_pd, min=free_bits)
    return kl_fb.sum(dim=1).mean()

def homology_loss(xls: torch.Tensor, xss: torch.Tensor) -> torch.Tensor:
    sum_term1 = xls * torch.log(xls / xss)
    sum_term2 = xss * torch.log(xss / xls)
    homology = ((sum_term1 + sum_term2).sum(dim=1) / 2).mean()
    return homology