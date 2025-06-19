"""
Run inference on a trained model and return the reconstructed spectra.
Extract the latent vectors.
"""
import sys
import os
import torch
import numpy as np
from typing import Literal, Union
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from mineral_analysis.endmember_extraction import extract_endmembers
from dimension_reduction.vae.vae import VAE  
from sklearn.preprocessing import MinMaxScaler
from dimension_reduction.ss_vae.dataloaders import get_dataloaders, get_dataloaders_ssvae
from dimension_reduction.ss_vae.spatial_spectral_vae import SpatialSpectralNet
from dimension_reduction.ss_vae.spectral_encoder import SpectralEncoder
from dimension_reduction.ss_vae.local_sensing import LocalSensingNet
from dimension_reduction.ss_vae.sequential_sensing import SequentialSensingNet
from dimension_reduction.ss_vae.config import get_config

config = get_config()

def load_model_state_dict(model_name: Literal['vae', 'ss-vae'], model_path: str, n_bands: int) -> Union[VAE, SpatialSpectralNet]:
    if model_name == 'ss-vae':
        state = torch.load(model_path, map_location='cpu', weights_only=False)
        raw_state_dict = state['model_state'] if 'model_state' in state else state

        # Remove '_orig_mod.' prefix from all keys
        cleaned_state_dict = {
            k.replace("_orig_mod.", ""): v
            for k, v in raw_state_dict.items()
        }
        model_ss = SpatialSpectralNet(
            n_bands=n_bands,  # number of spectral bands
            patch_size=state['config'].patch_size,  # patch size
            ld=state['config'].latent_dim,
            hidden_dim=state['config'].hidden_dim,
            lstm_layers=state['config'].lstm_layers,
            cnn_layers=state['config'].cnn_layers,
            free_bits=state['config'].free_bits,
        )
        model_ss.load_state_dict(cleaned_state_dict)
        return model_ss
    elif model_name == 'vae':
        state = torch.load(model_path, map_location='cpu', weights_only=False)
        raw_state_dict = state['model_state'] if 'model_state' in state else state

        # Remove '_orig_mod.' prefix from all keys
        cleaned_state_dict = {
            k.replace("_orig_mod.", ""): v
            for k, v in raw_state_dict.items()
        }
        model_vae = VAE(
            input_dim=n_bands,  # n_bands
            latent_dim=state['config'].latent_dim,
            hidden_dim=state['config'].hidden_dim,
        )
        model_vae.load_state_dict(cleaned_state_dict)
        return model_vae

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
    print(">>> >Running latest extract_latent_vectors")
     # Set the model to evaluation mode
    input_tensor = torch.from_numpy(input_data).float()
    if model_name == 'ss-vae':
        model_ss = load_model_state_dict(model_name, model_path, input_data.shape[1])
        model_ss.eval()  # Set the model to evaluation mode
        input_dl, _ = get_dataloaders_ssvae(data_path=config.data_path, batch_size=input_data.shape[0], neighborhood_size=5, test_size=0)
        for batch in input_dl:
            x=batch.float()
            # print(x.shape)
            with torch.inference_mode():
                sampled_mean, revised_mean, log_var = model_ss.encoder(x) #type: ignore
            print(revised_mean.shape)

        latent_vector = revised_mean.detach().numpy()
    elif model_name =='vae':
        model_vae = load_model_state_dict(model_name, model_path, input_data.shape[1])
        model_vae.eval()
        mean, log_var, _ = model_vae(input_tensor)
        latent_vector = mean.detach().numpy()  # Use the mean as the latent vector
    return latent_vector

def get_recon_spectra(model_name: Literal['vae', 'ss-vae'], model_path: str, input_data: np.ndarray) -> np.ndarray:
    """
    Get reconstructed spectra from a model given the input data.

    Parameters:
    - model_name (str): Name of the model to be used for inference.
    - model_path (str): Path to the entire model file.
    - input_data (np.ndarray): the pixel spectra (H*W, n_bands).

    Returns:
    - np.ndarray: Reconstructed spectra from the model.(H*W, n_bands)
    """
    print(">>> Running latest get_recon_spectra")
    input_tensor = torch.from_numpy(input_data).float()
    
    if model_name == 'ss-vae':
        model_ss = load_model_state_dict(model_name, model_path, input_data.shape[1])
        model_ss.eval()  # Set the model to evaluation mode
        input_dl, _ = get_dataloaders_ssvae(data_path=config.data_path, batch_size=input_data.shape[0], neighborhood_size=5, test_size=0)
        for batch in input_dl:
            x=batch.float()
            with torch.inference_mode():
                recon = model_ss(x)
        recon_np = recon.cpu().numpy()  # shape: (effective_rows*effective_cols, B)
    else:
        model_vae = load_model_state_dict(model_name, model_path, input_data.shape[1])
        model_vae.eval()
        with torch.inference_mode():
            _, _, recon = model_vae(input_tensor)
        recon_np = recon.detach().cpu().numpy()  # shape: (rows*cols, B)

    return recon_np