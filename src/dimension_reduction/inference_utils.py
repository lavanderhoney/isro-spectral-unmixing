"""
Run inference on a trained model and return the reconstructed spectra.
Extract the latent vectors.
"""
import torch
import numpy as np
from typing import Literal, Union
from argparse import Namespace
from matplotlib import pyplot as plt
from mineral_analysis.endmember_extraction import extract_endmembers
from dimension_reduction.vae.vae import VAE  
from dimension_reduction.ss_vae.dataloaders import get_dataloaders, get_dataloaders_ssvae
from dimension_reduction.ss_vae.spatial_spectral_vae import SpatialSpectralNet
from dimension_reduction.ss_vae.config import get_config

# config = get_config()

def load_model_state_dict(model_name: Literal['vae', 'ss-vae'], model_path: str, n_bands: int) -> Union[VAE, SpatialSpectralNet]:
    if model_name == 'ss-vae':
        state = torch.load(model_path, map_location='cpu', weights_only=False)
        raw_state_dict = state['model_state'] if 'model_state' in state else state

        # Remove '_orig_mod.' prefix from all keys
        cleaned_state_dict = {
            k.replace("_orig_mod.", ""): v
            for k, v in raw_state_dict.items()
        }
        # print(cleaned_state_dict.keys())
        # print(state.keys())
        model_ss = SpatialSpectralNet(
            n_bands=n_bands,  # number of spectral bands
            patch_size=state['config'].patch_size,  # patch size
            ld=4,
            hidden_dim=state['config'].hidden_dim,
            em_spectra=cleaned_state_dict['decoder.E'].detach().numpy(),  # type: ignore
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
        print(cleaned_state_dict.keys())
        model_vae = VAE(
            input_dim=n_bands,  # n_bands
            latent_dim=4,
            hidden_dim=state['config'].hidden_dim,
            em_spectra=cleaned_state_dict['E'].detach().numpy(),  # type: ignore
        )
        model_vae.load_state_dict(cleaned_state_dict)
        return model_vae

def extract_latent_vectors(model_name: Literal['vae', 'ss-vae'], model_path: str, input_data: np.ndarray, config: Namespace) -> np.ndarray:
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
            print("in ssvae :", x.shape)
            with torch.inference_mode():
                latent_vector, revised_mean, recon = model_ss(x) #type: ignore
            print("in ssvae: ", revised_mean.shape, latent_vector.shape)

        latent_vector = latent_vector.detach().numpy()
    elif model_name =='vae':
        model_vae = load_model_state_dict(model_name, model_path, input_data.shape[1])
        model_vae.eval()
        mean, log_var, _ = model_vae(input_tensor)
        
        # rather than using mean as latent vector, sample from the distribution, as I applied the softmax constraint on the sampled latent vector, not the mean
        latent_vector = model_vae.sample(mean, log_var).detach().numpy()  # type: ignore
        # latent_vector = mean.detach().numpy()  # Use the mean as the latent vector
    return latent_vector

def get_recon_spectra(model_name: Literal['vae', 'ss-vae'], model_path: str, input_data: np.ndarray, config: Namespace) -> np.ndarray:
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
        input_dl, _ = get_dataloaders_ssvae(data_path=config.data_path, batch_size=input_data.shape[0], neighborhood_size=config.patch_size, test_size=0)
        for batch in input_dl:
            x=batch.float()
            with torch.inference_mode():
                z, mean, recon = model_ss(x)
        if hasattr(recon, 'cpu'):
            recon = recon.cpu().numpy()  # shape: (effective_rows*effective_cols, B)
        recon_np = recon
    else:
        model_vae = load_model_state_dict(model_name, model_path, input_data.shape[1])
        model_vae.eval()
        with torch.inference_mode():
            _, _, recon = model_vae(input_tensor)
        if len(recon)==2:
            recon = recon[0] # by mistake, if vae returns a tuple of out, E_positive. Changed its fwd method to return out only now
        print("Reconstructed spectra shape:", recon.shape)
        recon_np = recon.detach().cpu().numpy()  # shape: (rows*cols, B)

    return recon_np