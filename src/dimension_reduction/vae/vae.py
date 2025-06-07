import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:   
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

        # Decoder layers
        self.dec_linear1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_ln1 = nn.LayerNorm(hidden_dim)
        self.dec_act1 = nn.LeakyReLU()
        self.dec_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dec_ln2 = nn.LayerNorm(hidden_dim)
        self.dec_act2 = nn.LeakyReLU()
        self.dec_linear3 = nn.Linear(hidden_dim, input_dim)

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
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # First block
        z = self.dec_linear1(z)
        z = self.dec_ln1(z)
        z = self.dec_act1(z)
        z_skip = z  # Save for skip connection
        
        # Second block with skip connection
        z = self.dec_linear2(z)
        z = self.dec_ln2(z)
        z = self.dec_act2(z)
        z = z + z_skip  # Add skip connection
        
        # Output layer
        out = self.dec_linear3(z)
        return out

    def forward(self, x):
        # Encoder part
        mean, log_var = self.encode(x)
        # Sampling
        z = self.sample(mean, log_var)
        # Decoder
        out = self.decode(z)
        return mean, log_var, out