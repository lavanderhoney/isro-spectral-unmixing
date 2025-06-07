"""
beta-VAE's encoder part for the spectral sensing.
Takes input of (batch_size, n_bands), which is the center pixel of the s*s patch.
Returns a mean vector of (batch_size, ld//2), and the log_var, i.e noise vector of (batch_size, ld). 
The latent vector is not sampled here, it will be sampled after revising this mean vector.

The encoder architecture is from the fc-vae, so when that is implemented from kaggle into this repo, i'll just import it from there
"""
import torch
import torch.nn as nn
from typing import Tuple

class SpectralEncoder(nn.Module):
    def __init__(self, n_bands:int, ld:int, hidden_dim: int) -> None:
        super().__init__()
        self.n_bands = n_bands
        self.ld = ld
        self.hidden_dim = hidden_dim
        # Encoder layers
        self.enc_linear1 = nn.Linear(in_features=self.n_bands, out_features=self.hidden_dim)
        self.enc_ln1 = nn.LayerNorm(self.hidden_dim)
        self.enc_act1 = nn.LeakyReLU()
        self.enc_linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.enc_ln2 = nn.LayerNorm(self.hidden_dim)
        self.enc_act2 = nn.LeakyReLU()
        self.enc_linear3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.enc_ln3 = nn.LayerNorm(self.hidden_dim)
        self.enc_act3 = nn.LeakyReLU()
        
        self.mean_fc = nn.Linear(self.hidden_dim, self.ld//2)
        self.log_var = nn.Linear(self.hidden_dim, self.ld)

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

    # def sample(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    #     std = torch.exp(0.5 * log_var)
    #     epsilon = torch.randn_like(std)
    #     z = std * epsilon + mean
    #     return z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encoder part
        mean, log_var = self.encode(x)
        # z = self.sample(mean, log_var)
        return mean, log_var