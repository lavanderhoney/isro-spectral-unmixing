"""
Revises the mean vector from spectral encoder with local sensing and sequential sensing information.
Samples the latent vector ld from the revised mean and noise, and reconstructs it with the decoder.
The decoder architecture is from my fc-vae.
This file will include the complete forward pass of the model.
"""
# %%
import torch
import torch.nn as nn
from typing import Tuple
from .spectral_encoder import SpectralEncoder
from .local_sensing import LocalSensingNet
from .sequential_sensing import SequentialSensingNet
from .utils import extract_sequential_data, extract_spectral_data
from .loss import homology_loss, kl_loss, reconstruction_loss

class SpatialSpectralEncoder(nn.Module):
    def __init__(self, n_bands: int, patch_size: int, ld: int, hidden_dim: int, lstm_layers: int=3, cnn_layers: int =3, free_bits: float =0.0):
        super().__init__()
        self.n_bands = n_bands
        self.s = patch_size
        self.ld = ld
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.cnn_layers = cnn_layers
        self.free_bits = free_bits
        
        self.spectral_encoder = SpectralEncoder(self.n_bands, self.ld, self.hidden_dim)
        self.local_spatial_sensing = LocalSensingNet(self.n_bands, self.s, self.ld, self.cnn_layers)
        self.sequential_spatial_sensing = SequentialSensingNet(self.n_bands, self.s, self.ld, self.lstm_layers)
        
        self.homology_loss_term: torch.Tensor
        self.kl_loss_term: torch.Tensor

    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        forward pass through all three nets, and return the sampled, concatenated latent vector.\n
        also computes the homology and kl loss terms \n
        x: un-transformed input shape from the general DL, i.e, (batch_size, s, s, B)
        """
        
        seq_sensing_data = extract_sequential_data(x) #
        spec_sensing_data = extract_spectral_data(x)
        loc_sensing_data = x
        
        x_mean, log_var = self.spectral_encoder(spec_sensing_data) # (batch, ld//2), (batch,ld)
        x_ls = self.local_spatial_sensing(loc_sensing_data) #(batch, ld//4)
        x_ss = self.sequential_spatial_sensing(seq_sensing_data) #(batch, ld//4)
        
        revised_mean = torch.concat((x_ls, x_ss, x_mean), 1) #as perp
        
        #calc the homology loss and KL loss for use in training loop
        self.homology_loss_term = homology_loss(x_ls, x_ss)
        self.kl_loss_term = kl_loss(revised_mean, log_var, self.free_bits)
        
        #sample the latent vector from revised_mean and log_var
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        latent_vector = std * epsilon + revised_mean # (batch, ld)
        
        return latent_vector, revised_mean, log_var

class SpatialSpectralDecoder(nn.Module):
    def __init__(self, n_bands:int, ld: int, hidden_dim: int) -> None:
        super().__init__()
        self.n_bands = n_bands
        self.ld = ld
        self.hidden_dim = hidden_dim
        
         # Decoder layers
        self.dec_linear1 = nn.Linear(self.ld, self.hidden_dim)
        self.dec_ln1 = nn.LayerNorm(self.hidden_dim)
        self.dec_act1 = nn.LeakyReLU()
        self.dec_linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dec_ln2 = nn.LayerNorm(self.hidden_dim)
        self.dec_act2 = nn.LeakyReLU()
        self.dec_linear3 = nn.Linear(self.hidden_dim, self.n_bands)
      
    def decode(self, z):
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
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        takes in the latent_vector from the encoder, and returns the reconstructed pixel spectra 
        """
        recon_spectra = self.decode(x)
        return recon_spectra
    
class SpatialSpectralNet(nn.Module):
    def __init__(self, n_bands: int, patch_size: int, ld: int, hidden_dim: int, lstm_layers: int=3, cnn_layers: int =3, free_bits: float =0.0):
        super().__init__()
        self.spectral_bands = n_bands
        self.encoder = SpatialSpectralEncoder(n_bands, patch_size, ld, hidden_dim, lstm_layers, cnn_layers, free_bits)
        self.decoder = SpatialSpectralDecoder(n_bands, ld, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, s, s, B)
        """
        z, mean, log_var = self.encoder(x)
        return self.decoder(z)
    
    
# %%

# #testing
# x = torch.randn(32, 5, 5, 109)
# ld=12
# net = SpatialSpectralNet(x.shape[-1], x.shape[1],ld, hidden_dim=64)
# out = net(x)
# print("output: ", out.shape)
# %%
