#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Literal
#%%
class AE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, em_spectra, constraint: Literal['softmax', 'man'] = 'softmax', scaling: float = 0.8) -> None:
        super().__init__()
        # Encoder layers
        self.enc_linear1 = nn.Linear(input_dim, hidden_dim) #109,64
        # self.enc_ln1 = nn.LayerNorm(hidden_dim)
        self.enc_bn1 = nn.BatchNorm1d(hidden_dim)
        self.enc_act1 = nn.LeakyReLU()

        self.enc_linear2 = nn.Linear(hidden_dim, hidden_dim//2) #64,32
        self.enc_bn2 = nn.BatchNorm1d(hidden_dim//2)
        self.enc_act2 = nn.LeakyReLU()

        self.enc_linear3 = nn.Linear(hidden_dim//2, hidden_dim//4) #32,16
        self.enc_bn3 = nn.BatchNorm1d(hidden_dim//4)
        self.enc_act3 = nn.LeakyReLU()
        # Encoder's final layer to produce latent_dim (M) outputs
        self.encoder_output_layer = nn.Linear(hidden_dim//4, latent_dim) # 16,4
        # Learnable or Fixed Endmember Matrix E
        # latent_dim is M (number of endmembers)
        # input_dim is N (number of spectral bands)
        E_tensor = torch.from_numpy(em_spectra).float()
        self.E = nn.Parameter(E_tensor, requires_grad=False)  # M x N matrix
        self.constraint = constraint  # 'softmax' or 'man'(i.e, manual normalization)
        self.beta = scaling

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # First block
        x = self.enc_linear1(x)
        x = self.enc_bn1(x)
        x = self.enc_act1(x)
        
        x = self.enc_linear2(x)
        x = self.enc_bn2(x)
        x = self.enc_act2(x)
      
        x = self.enc_linear3(x)
        x = self.enc_bn3(x)
        x = self.enc_act3(x)

        z = self.encoder_output_layer(x)
        
        #----- ANC and ASC constraints -----
        if self.constraint == 'softmax':
            z = F.softmax(z/self.beta, dim=1)  # too aggressive
        elif self.constraint == 'man':
            z = F.relu(z)  # ANC
            z = z / (torch.sum(z, dim=1, keepdim=True) + 1e-6)  # ASC: Normalize to sum to 1
        else:
            raise ValueError("Invalid constraint type. Use 'softmax' or 'man'.")
        return z  # (batch_size, M)

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Linear Mixture Model
        # z: (batch_size, M)
        # E: (M, N)
        # Output: (batch_size, N)
        E_positive = F.relu(self.E)
        x_hat = torch.matmul(z, E_positive)  # z (batch_size, M) * E (M, N) -> x_hat (batch_size, N)
        # x_hat = F.relu(x_hat)  # Ensure non-negativity of the output
        return x_hat, E_positive
      

    def forward(self, x):
        # Encoder part
        z = self.encode(x)
        x_hat, E_positive = self.decode(z)
        return x_hat, z, E_positive

#loss functions
def spectral_angle_distance_loss(x_hat: torch.Tensor, x: torch.Tensor, eps:float = 1e-8) -> torch.Tensor:
    assert not torch.isnan(x_hat).any(), "NaN in x_hat"
    assert not torch.isnan(x).any(), "NaN in x"

    # x_hat, x = F.softplus(x_hat), F.softplus(x)  # Ensure non-negativity

    cos_theta = F.cosine_similarity(x_hat, x, dim=1, eps=eps)  # Compute cosine similarity
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # Ensure values are in the valid range for acos
    angle = torch.acos(cos_theta)
    return torch.mean(angle) #average over the batch

def spectral_information_divergence_loss(x_hat: torch.Tensor, x: torch.Tensor, eps:float = 1e-8) -> torch.Tensor:
    """
    Computes the Spectral Information Divergence (SID) loss between the reconstructed and original spectra.
    
    SID(p, q) = D_KL(p || q) + D_KL(q || p)

    where p and q are normalized versions of x and x_hat, and D_KL is the
    Kullback-Leibler divergence.

    Args:
        x_hat (torch.Tensor): Reconstructed spectra, shape (batch_size, num_bands).
        x (torch.Tensor): Original spectra, shape (batch_size, num_bands).
        epsilon (float): Small value for numerical stability to avoid log(0)
                         when dividing by zero or taking log of zero.

    Returns:
        torch.Tensor: The average SID loss over the batch.
    """
    x_hat, x = x_hat + eps, x + eps  # Add epsilon to avoid log(0)
    p = x/torch.sum(x, dim=-1, keepdim=True)
    q = x_hat/torch.sum(x_hat, dim=-1, keepdim=True)

    kl_pq = torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)
    kl_qp = torch.sum(q * (torch.log(q) - torch.log(p)), dim=-1)
    sid = kl_pq + kl_qp
    return torch.mean(sid)  # Average over the batch
    
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

    ones_M = torch.eye(M, 1, device=E.device)
    I_r = torch.eye(M, device=E.device, dtype=E.dtype)
    P = I_r - 1/r * (ones_M @ ones_M.T) # M x M centering matrix

    # Compute the Total Variation term
    TV = torch.norm(torch.matmul(E.T,P), p='fro') ** 2  # Frobenius norm squared
    return TV

def entropy_loss(z: torch.Tensor) -> torch.Tensor:
    """
    Computes the entropy loss for the latent abundances z.
    Args:
        z (torch.Tensor): Latent abundances, shape (batch_size, M).
    Returns:    
        torch.Tensor: The average entropy loss over the batch.
    """
    # Compute entropy for each sample in the batch
    entropy = -torch.sum(z * torch.log(z + 1e-8), dim=1)  # Add small value for numerical stability
    return torch.mean(entropy)  # Average over the batch

if __name__ == "__main__":
    input_dim = 10  # Number of spectral bands
    # hidden_dim = 64
    # latent_dim = 4  # Number of endmembers

    # model = AE(input_dim, hidden_dim, latent_dim)
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.shape}")

    # # Example input tensor
    # x = torch.randn(5, input_dim)  # Batch size of 5
    # x_hat, z, E_positive = model(x)
    
    # print("Reconstructed Output:", x_hat)
    # print("Latent Abundances (z):", z)
    # print("Positive Endmember Matrix (E):", E_positive)