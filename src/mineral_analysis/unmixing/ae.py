#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple
#%%
class AE(nn.Module):
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
        # Encoder's final layer to produce latent_dim (M) outputs
        self.encoder_output_layer = nn.Linear(hidden_dim, latent_dim) # From last hidden_dim to M
        # Learnable Endmember Matrix E
        # latent_dim is M (number of endmembers)
        # input_dim is N (number of spectral bands)
        self.E = nn.Parameter(torch.randn(latent_dim, input_dim))  # M x N matrix

    def encode(self, x: torch.Tensor) -> torch.Tensor:
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
        # Final layer to get the latent abundances 'z'
        z = self.encoder_output_layer(x)
        z = F.relu(z) # Ensures non-negative abundances *from the encoder*
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Linear Mixture Model
        # z: (batch_size, M)
        # E: (M, N)
        # Output: (batch_size, N)
        E_positive = F.relu(self.E)
        x_hat = torch.matmul(z, E_positive)  # z (batch_size, M) * E (M, N) -> x_hat (batch_size, N)
        x_hat = F.relu(x_hat)  # Ensure non-negativity of the output
        return x_hat
      

    def forward(self, x):
        # Encoder part
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z, F.relu(self.E)

#loss functions
def spectral_angle_distance_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    x_hat, x = F.relu(x_hat), F.relu(x)
    dot_product = torch.inner(x_hat, x)
    norms = torch.norm(x_hat, dim=1) * torch.norm(x, dim=1)
    cos_theta = dot_product / norms
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # Ensure values are in the valid range for acos
    angle = torch.acos(cos_theta)
    return torch.mean(angle) #average over the batch

def total_variation(E: torch.Tensor) -> torch.Tensor:
    """
    Compute the Total Variation function TV_r(E) for the endmember matrix E. Acts as a minimum volume regularizer term
    
    Parameters:
    - E: torch.tensor of shape (m, n), where m is the number of endmembers and n is bands.
    - r: scalar parameter (default is n, the number of columns in E).
    
    Returns:
    - TV: a scalar torch.tensor representing the Total Variation value ||E (I_n - (1/r) 1_n 1_n^T)||_F^2.
    """
    M, N = E.shape
    r = M # r is the number of endmembers as per the paper

    # Construct the matrix P = I_M - (1/r) 1_M 1_M^T
    ones_M = torch.eye(M, 1, device=E.device)
    I_r = torch.eye(M, device=E.device, dtype=E.dtype)
    P = I_r - 1/r * (ones_M @ ones_M.T) # M x M centering matrix

    # Compute the Total Variation term
    TV = torch.norm(torch.matmul(E.T,P), p='fro') ** 2  # Frobenius norm squared
    return TV
