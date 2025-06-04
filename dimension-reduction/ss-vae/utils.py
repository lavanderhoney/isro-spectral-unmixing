"""
Utility functions for the SS-VAE model.
"""
import numpy as np
import torch

# these extract functions will be called during fwd method, so they take the batch as input
def extract_sequential_data(batch_data: torch.Tensor) -> torch.Tensor:
    """
    bathch_data: (batch_size, s, s, B) \n
    make sequences of length s^2 and return shape (batch_size, s^2, B)
    """
    batch_size, s, _, B = batch_data.shape
    sequenced_data = batch_data.reshape(batch_size, s * s, B)  # reshape to (batch_size, s^2, B)
    return sequenced_data

def extract_spectral_data(batch_data: torch.Tensor) -> torch.Tensor:
    """
    returns the spectra of the centeral  pixel in each patch, i.e (batch_size, B, 1)
    """
    batch_size, s, _, B = batch_data.shape
    return batch_data[:, s // 2, s // 2, :]  # shape (batch_size, B)
