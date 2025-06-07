"""
takes input the model path, run inference, and returns the latent vectors.
The function will istantiate the model, load the weights, and run inference on the input data.
Then use functions from other modules on it
"""
import sys
import os

import torch
import numpy as np
from typing import Literal 
from mineral_analysis.endmember_extraction import extract_endmembers
from .ss_vae.spatial_spectral_vae import SpatialSpectralNet
# from vae.vae import VAE
def extract_latent_vectors(model_name: Literal['vae', 'ss-vae'], model_path: str, input_data: np.ndarray) -> np.ndarray:
    """
    Extract latent vectors from a model given the input data.

    Parameters:
    - model_name (str): Name of the model to be used for inference.
    - model_path (str): Path to the entire model file.
    - input_data (np.ndarray): the pixel spectra (H*W, n_bands).

    Returns:
    - np.ndarray: Latent vectors extracted from the model.
    """
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()  # Set the model to evaluation mode
    
    if model_name == 'ss-vae':
        latent_vector = model.encoder(input_data)
        
    return latent_vector.detach().numpy()  # Convert to numpy array and detach from the computation graph
   
def extract_endmembers_from_latent(latent_vectors: np.ndarray, wavelengths: np.ndarray, algorithm: Literal['nfindr', 'vca', 'fippi', 'atgp'], rows: int, cols:int, n_endmembers: int = 4, ) -> np.ndarray:
    """    Extract endmembers from latent vectors using specified algorithm.
    Parameters:
    - latent_vectors: (n_samples, n_bands) array of latent vectors.
    - wavelengths: (n_bands,) array of wavelength values.
    - algorithm: 'nfindr', 'vca', 'fippi', or 'atgp'.
    - n_endmembers: number of endmembers to extract.
    Returns:
    - endmembers: (n_endmembers, n_bands) array of extracted endmember spectra.
    """
    # Reshape latent vectors to match the expected input shape for endmember extraction
    H_t_latent = latent_vectors.reshape(rows, cols, -1)  # Assuming latent_vectors is (H*W, n_bands)
    
    # Call the endmember extraction function
    endmembers, abundance_maps = extract_endmembers(H_t_latent, wavelengths, algorithm, n_endmembers=n_endmembers, show_endmembers=True, show_amaps=True)
    
    return endmembers  # Return only the endmembers
    
if __name__ == "__main__":
    data_path = "data/reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz"
    ssvae_path = "models/ss-vae/ss_vae_model.pth"
    
    data = np.load(data_path)
    data_cube = data['refl_data']
    wavelengths = data['wavelengths']
    n_bands, rows, cols = data_cube.shape

    H_t = data_cube.transpose(1, 2, 0) # Transpose to (rows, cols, bands)
    input_data = H_t.reshape(-1, n_bands)  # Reshape to (H*W, n_bands)
    
    # Extract latent vectors using the ss-vae model
    latent_vectors = extract_latent_vectors('ss-vae', ssvae_path, input_data)
    print("Latent vectors shape:", latent_vectors.shape)
    
    # Extract endmembers from the latent vectors
    endmembers = extract_endmembers_from_latent(latent_vectors, wavelengths, algorithm='nfindr', rows=rows, cols=cols, n_endmembers=4, )
    print("Endmembers shape:", endmembers.shape)
    print("Endmembers:", endmembers)