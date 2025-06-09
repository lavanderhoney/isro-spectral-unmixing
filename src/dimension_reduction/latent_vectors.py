"""
takes input the model path, run inference, and returns the latent vectors.
The function will istantiate the model, load the weights, and run inference on the input data.
Then use functions from other modules on it
"""
#%%
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
from dimension_reduction.ss_vae.dataloaders import get_dataloaders
from dimension_reduction.ss_vae.spatial_spectral_vae import SpatialSpectralNet
from dimension_reduction.ss_vae.spectral_encoder import SpectralEncoder
from dimension_reduction.ss_vae.local_sensing import LocalSensingNet
from dimension_reduction.ss_vae.sequential_sensing import SequentialSensingNet

def load_model_state_dict(model_name: Literal['vae', 'ss-vae'], model_path: str) -> Union[VAE, SpatialSpectralNet]:
    if model_name == 'ss-vae':
        state = torch.load(model_path, map_location='cpu', weights_only=False)
        raw_state_dict = state['model_state'] if 'model_state' in state else state

        # Remove '_orig_mod.' prefix from all keys
        cleaned_state_dict = {
            k.replace("_orig_mod.", ""): v
            for k, v in raw_state_dict.items()
        }
        model_ss = SpatialSpectralNet(
            n_bands=input_data.shape[1],  # number of spectral bands
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
            input_dim=input_data.shape[1],  # n_bands
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
        model_ss = load_model_state_dict(model_name, model_path)
        model_ss.eval()  # Set the model to evaluation mode
        input_dl, _ = get_dataloaders(batch_size=input_data.shape[0], neighborhood_size=5, test_size=0)
        for batch in input_dl:
            x=batch.float()
            # print(x.shape)
            with torch.inference_mode():
                sampled_mean, revised_mean, log_var = model_ss.encoder(x) #type: ignore
            print(revised_mean.shape)

        latent_vector = revised_mean.detach().numpy()
    elif model_name =='vae':
        model_vae = load_model_state_dict(model_name, model_path)
        model_vae.eval()
        mean, log_var, _ = model_vae(input_tensor)
        latent_vector = mean.detach().numpy()  # Use the mean as the latent vector
    return latent_vector

def show_recon_image(
    model_name: Literal['vae', 'ss-vae'],
    model_path: str,
    input_data: np.ndarray,
    rows: int,
    cols: int,
    n_samples: int = 3
) -> None:
    """
    Show reconstructed images from the model given the input data, and compute SAM.

    Parameters:
    - model_name (str): 'vae' or 'ss-vae'.
    - model_path (str): Path to the model or state dict.
    - input_data (np.ndarray): Pixel spectra array of shape (H*W, B).
    - rows (int), cols (int): Original cube dimensions.
    - n_samples (int): Number of random spectra to plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    from typing import Literal
    import torch

    print(">>> Running show_recon_image with updated plotting and normalization")
    band_index = 30  # spectral band to visualize

    # Load and run model
    if model_name == 'ss-vae':
        # Get full input patches loader to reconstruct every pixel
        input_dl, _ = get_dataloaders(batch_size=input_data.shape[0], neighborhood_size=5, test_size=0)
        model_ss = load_model_state_dict(model_name, model_path)
        model_ss.eval()
        for batch in input_dl:
            x = batch.float()
            with torch.inference_mode():
                recon = model_ss(x)
        recon_np = recon.cpu().numpy()  # shape: (effective_rows*effective_cols, B)
        effective_rows, effective_cols = 997, 246
    else:
        model_vae = load_model_state_dict(model_name, model_path)
        model_vae.eval()
        with torch.inference_mode():
            _, _, recon = model_vae(torch.from_numpy(input_data).float())
        recon_np = recon.detach().cpu().numpy()  # shape: (rows*cols, B)
        effective_rows, effective_cols = rows, cols

    # Reshape input for plotting
    cube = input_data.reshape(rows, cols, -1)
    if model_name == 'ss-vae':
        # Center-crop original cube to match latent output size
        start_r = (rows - effective_rows) // 2
        start_c = (cols - effective_cols) // 2
        cube = cube[start_r:start_r + effective_rows, start_c:start_c + effective_cols, :]
    orig_band = cube[:, :, band_index]

    # Extract reconstructed band and normalize both for display
    recon_band = recon_np[:, band_index].reshape(effective_rows, effective_cols)
    orig_norm = (orig_band - orig_band.min()) / (orig_band.max() - orig_band.min() + 1e-6)
    recon_norm = (recon_band - recon_band.min()) / (recon_band.max() - recon_band.min() + 1e-6)

    # Plot side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    im1 = ax1.imshow(orig_norm, cmap='gray', vmin=0, vmax=1)
    ax1.set_title("Original Band 30")
    ax1.axis('off')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(recon_norm, cmap='gray', vmin=0, vmax=1)
    ax2.set_title("Reconstructed Band 30")
    ax2.axis('off')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.suptitle("Spectral Band Comparison (Normalized)")
    plt.tight_layout()
    out_path = Path(f"{model_name}_spectral_band_comparison.png")
    plt.savefig(out_path)
    plt.close()

    # Compute SAM over all pixels
    def compute_sam(X1, X2):
        dot = np.sum(X1 * X2, axis=1)
        norm1 = np.linalg.norm(X1, axis=1)
        norm2 = np.linalg.norm(X2, axis=1)
        cos_theta = dot / (norm1 * norm2 + 1e-6)
        angles = np.arccos(np.clip(cos_theta, -1, 1))
        return np.degrees(angles)

    flat_orig = cube.reshape(-1, cube.shape[2])
    sam_scores = compute_sam(flat_orig, recon_np)
    avg_sam = np.mean(sam_scores)
    print(f">>> Average SAM over all pixels: {avg_sam:.2f}°")

    # Plot random spectral profiles
    if n_samples > 0:
        idx = np.random.choice(flat_orig.shape[0], size=n_samples, replace=False)
        for i in idx:
            orig_spec = flat_orig[i]
            recon_spec = recon_np[i]
            # normalize per-spectrum for visibility
            orig_s = (orig_spec - orig_spec.min()) / (orig_spec.max() - orig_spec.min() + 1e-6)
            recon_s = (recon_spec - recon_spec.min()) / (recon_spec.max() - recon_spec.min() + 1e-6)
            plt.plot(orig_s, '--', label='Original')
            plt.plot(recon_s, '-', label='Reconstructed', alpha=0.7)
            plt.title(f"Pixel {i} Spectra (Normalized)")
            plt.xlabel("Band index")
            plt.ylabel("Normalized intensity")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"pixel_{i}_spectra_comparison_{model_name}.png")
            plt.close()

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
    print("H_t_latent shape:", H_t_latent.shape)
    wavelengths = np.arange(start=1, stop=H_t_latent.shape[2] + 1)  # Create a dummy wavelength array 
    # Call the endmember extraction function
    endmembers, abundance_maps = extract_endmembers(H_t_latent, wavelengths, algorithm, n_endmembers=n_endmembers, show_endmembers=True, show_amaps=True)
    
    return endmembers  # Return only the endmembers
    
if __name__ == "__main__":
    data_path = "/teamspace/studios/this_studio/isro-spectral-unmixing/data/den_reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz"
    ss_vae_path = "models/model_state_ss_vae_0609_035249.pth"
    vae_path = "models/model_state_vae_0609_045053.pth"
    
    data = np.load(data_path)
    data_cube = data['den_refl_data']
    wavelengths = data['wavelengths']
    n_bands, rows, cols = data_cube.shape

    H_t = data_cube.transpose(1, 2, 0) # Transpose to (rows, cols, bands)
    input_data = H_t.reshape(-1, n_bands) # Reshape to (H*W, n_bands)
    # Extract latent vectors using the ss-vae model
    # latent_vectors = extract_latent_vectors('vae', vae_path, input_data)
    # print("Latent vectors shape:", latent_vectors.shape)
    
    # Extract endmembers from the latent vectors
    # endmembers = extract_endmembers_from_latent(latent_vectors, wavelengths, algorithm='nfindr', rows=rows, cols=cols, n_endmembers=4, )
    # print("Endmembers shape:", endmembers.shape)
    # print("Endmembers:", endmembers)
    
    # Model's sanity check
    # show_recon_image('ss-vae', ss_vae_path, input_data, n_samples=3)
    show_recon_image('vae', vae_path, input_data,rows, cols, n_samples=3)
# %%