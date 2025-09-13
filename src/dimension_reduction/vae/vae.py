import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
from torch.nn import functional as F
class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, em_spectra: np.ndarray, constraint: str = "softmax", scaling: float = 1.5) -> None:   
        super().__init__()
        
        # Encoder layers
        self.enc_linear1 = nn.Linear(input_dim, hidden_dim)
        self.enc_ln1 = nn.LayerNorm(hidden_dim)
        self.enc_act1 = nn.LeakyReLU()
        self.enc_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_ln2 = nn.LayerNorm(hidden_dim)
        self.enc_act2 = nn.LeakyReLU()
        self.enc_linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_ln3 = nn.LayerNorm(hidden_dim)
        self.enc_act3 = nn.LeakyReLU()
        self.mean_fc = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

        self.constraint = constraint
        self.beta = scaling
        E_tensor = torch.from_numpy(em_spectra).float()
        self.E = nn.Parameter(E_tensor, requires_grad=False)
  
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # First block
        x = self.enc_linear1(x)
        x = self.enc_ln1(x)
        x = self.enc_act1(x)
        x_skip = x  # Save for skip connection
        
        # Second block with skip connection
        x = self.enc_linear2(x)
        x = self.enc_ln2(x)
        x = self.enc_act2(x)
        x = x + x_skip  # Add skip connection
        x_skip = x  # Update skip for next connection
        
        # Third block with skip connection
        x = self.enc_linear3(x)
        x = self.enc_ln3(x)
        x = self.enc_act3(x)
        x = x + x_skip  # Add skip connection
        
        # Compute mean and log variance
        mean = self.mean_fc(x)
        log_var = self.log_var(x)
        return mean, log_var

    def sample(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = std * epsilon + mean
        
        # put constraints on the sampled latent vector.
        z = F.softmax(z/self.beta, dim=1)  
        return z

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # ---- Unimixing style dimension reduction, so decoder is VCA's EMs ---------
        # # First block
        # z = self.dec_linear1(z)
        # z = self.dec_ln1(z)
        # z = self.dec_act1(z)
        # z_skip = z  # Save for skip connection
        
        # # Second block with skip connection
        # z = self.dec_linear2(z)
        # z = self.dec_ln2(z)
        # z = self.dec_act2(z)
        # z = z + z_skip  # Add skip connection
        
        # # Output layer
        # out = self.dec_linear3(z)
        # return out
        E_positive = F.relu(self.E)
        x_hat = torch.matmul(z, E_positive)  # z (batch_size, M) * E (M, N) -> x_hat (batch_size, N)
        # x_hat = F.relu(x_hat)  # Ensure non-negativity of the output
        return x_hat, E_positive

    def forward(self, x):
        """
        Forward pass of the VAE.
        Returns:
        - mean: Mean of the latent space
        - log_var: Log variance of the latent space
        - x_hat: Reconstructed input
        """
        #NOTE: Not in use in the training script (train_vae.py), but used in inference_utils.py
        # not returning the sampled latent vector is pretty dumb and should change it. Not following this in SS-VAE
        
        # Encoder part
        mean, log_var = self.encode(x)
        # Sampling
        z = self.sample(mean, log_var)
        # Decoder
        out, E_positive = self.decode(z)
        return mean, log_var, out