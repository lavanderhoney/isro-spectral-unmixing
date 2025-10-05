"""
Run inference on a trained model and return the reconstructed spectra.
Extract the latent vectors.
"""
import torch
import numpy as np
from typing import Literal, Union, Optional
from argparse import Namespace
from matplotlib import pyplot as plt
from mineral_analysis.endmember_extraction import extract_endmembers
from dimension_reduction.vae.vae import VAE  
from dimension_reduction.ss_vae.dataloaders import get_dataloaders, get_dataloaders_ssvae, samson_dataloader
from dimension_reduction.ss_vae.spatial_spectral_vae import SpatialSpectralNet
from dimension_reduction.ss_vae.config import get_config
from dimension_reduction.ss_vae.dataloaders import open_datacube
from mineral_analysis.unmixing.ae import AE, spectral_angle_distance_loss, mse_loss, spectral_information_divergence_loss, entropy_loss
from mineral_analysis.unmixing.ae2 import AE2
from mineral_analysis import endmember_extraction as eea

# config = get_config()

def load_model_state_dict(model_name: Literal['ae','ae2', 'vae', 'ss-vae'], model_path: str, n_bands: int,  data_path:str, em_path: Optional[str]=None,) -> Union[AE2, AE, VAE, SpatialSpectralNet]:
    state = torch.load(model_path, map_location='cpu', weights_only=False)
    raw_state_dict = state['model_state'] if 'model_state' in state else state
    # Remove '_orig_mod.' prefix from all keys
    cleaned_state_dict = {
        k.replace("_orig_mod.", ""): v
        for k, v in raw_state_dict.items()
    }
    # print("STATE DICT  :", cleaned_state_dict.keys())
    if model_name == 'ss-vae':
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
        print(cleaned_state_dict.keys())
        model_vae = VAE(
            input_dim=n_bands,  # n_bands
            latent_dim=4,
            hidden_dim=state['config'].hidden_dim,
            em_spectra=cleaned_state_dict['E'].detach().numpy(),  # type: ignore
        )
        model_vae.load_state_dict(cleaned_state_dict)
        return model_vae
    elif model_name == 'ae':
        # Extract endmember spectra if present in the state dict, otherwise set to None or handle accordingly
        if em_path:
            ems = np.load(em_path)
        else:
            if data_path is None:
                raise ValueError("data_path must be provided if em_path is not given.")
            H, wavelengths = open_datacube(data_path)
            H_t = H.transpose(1, 2, 0)
            ems, _ = eea.extract_endmembers(H_t, wavelengths, algorithm='vca', n_endmembers=4, show_endmembers=False, show_amaps=False)
        model_ae = AE(
            input_dim=n_bands,  # Number of spectral bands
            hidden_dim=state['config'].hidden_dim,
            latent_dim=4,
            em_spectra=ems,
            scaling=state['scaling'],  
        )
        model_ae.load_state_dict(cleaned_state_dict)
        model_ae.eval()
        print(model_ae.beta, model_ae.constraint)
        return model_ae 
    elif model_name == 'ae2':
        model = AE2(input_dim=156, hidden_dim=64, latent_dim=3) 
        model.load_state_dict(cleaned_state_dict)
        model.eval()
        return model


def extract_latent_vectors(model_name: Literal['ae', 'ae2', 'vae', 'ss-vae'], model_path: str, config: Namespace, data_path:str, em_path: Optional[str]=None) -> np.ndarray:
    """
    Extract latent vectors from a model given the input data.

    Parameters:
    - model_name (str): Name of the model to be used for inference.
    - model_path (str): Path to the entire model file.
    - config (Namespace): Configuration object containing model parameters.
    - data_path (str, optional): Path to the input data file. Required for 'ss-vae' and 'ae' models.
    - em_path (str, optional): Path to the endmember spectra file. Required for 'ae' model.
    Returns:
    - np.ndarray: Latent vectors extracted from the model.
    """
    print(">>> >Running latest extract_latent_vectors")
     # Set the model to evaluation mode
    if model_name == 'ss-vae':
        model_ss = load_model_state_dict(model_name, model_path, n_bands=109, data_path=data_path)
        model_ss.eval()  # Set the model to evaluation mode
        input_dl, _ = get_dataloaders_ssvae(data_path=data_path, neighborhood_size=5, test_size=0)
        for batch in input_dl:
            x=batch.float()
            print("in ssvae :", x.shape)
            with torch.inference_mode():
                latent_vector, revised_mean, recon = model_ss(x) #type: ignore
            print("in ssvae: ", revised_mean.shape, latent_vector.shape)

        latent_vector = latent_vector.detach().numpy()
    elif model_name =='vae':
        model_vae = load_model_state_dict(model_name, model_path, n_bands=109, data_path=data_path)
        model_vae.eval()
        input_dl, _, _ = get_dataloaders(data_path=data_path, batch_size=32, test_size=0.0)
        latent_vecs = []
        entropy_losses = []
        for batch in input_dl:
            x = batch[0].float()
            with torch.inference_mode():
                mean, log_var, out = model_vae(x)
            # rather than using mean as latent vector, sample from the distribution, as I applied the softmax constraint on the sampled latent vector, not the mean
            latent_vecs.append(model_vae.sample(mean, log_var).detach().numpy())  # type: ignore
            entropy_loss_term = entropy_loss(torch.softmax(mean, dim=-1))  # Apply softmax to mean before computing entropy
            entropy_losses.append(entropy_loss_term.item())
        print(f"Average Entropy Loss: {np.mean(entropy_losses)}")
        latent_vector = np.concatenate(latent_vecs, axis=0)
        # latent_vector = mean.detach().numpy()  # Use the mean as the latent vector
    elif model_name == 'ae':
        model_ae = load_model_state_dict(model_name, model_path, n_bands=109, data_path=data_path, em_path=em_path)
        input_dl, _, _ = get_dataloaders(data_path=data_path, batch_size=32, test_size=0.0)
        # print("NOTE: data path fetched from config")
        abundance_vectors = []
        entropy_losses = []
        for x in input_dl:
            x = x[0].float()  # type: ignore # Extract the tensor from the tuple
            with torch.inference_mode():
                x_hat, z, E_positive = model_ae(x)
            # print(f"Reconstruction Loss: {loss.item()}")
            abundance_vectors.append(z.detach().numpy())
            entropy_loss_term = entropy_loss(z)
            entropy_losses.append(entropy_loss_term.item())
        print(f"Average Entropy Loss: {np.mean(entropy_losses)}")
        latent_vector = np.concatenate(abundance_vectors, axis=0)
    elif model_name == 'ae2':
        model = load_model_state_dict(model_name, model_path, n_bands=156, data_path=data_path)
        input_dl, _ = samson_dataloader(data_path=data_path, batch_size=20, test_size=0.0)
        abundance_vectors = []
        for x in input_dl:
            x = x[0].float()  # type: ignore # Extract the tensor from the tuple
            with torch.inference_mode():
                x_hat, z = model(x)
            abundance_vectors.append(z.detach().numpy())
        latent_vector = np.concatenate(abundance_vectors, axis=0)
    return latent_vector

def get_recon_spectra(model_name: Literal['ae', 'ae2', 'vae', 'ss-vae'], model_path: str, config: Namespace, data_path: str, em_path: Optional[str]=None) -> np.ndarray:
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
    # input_tensor = torch.from_numpy(input_data).float()
    
    if model_name == 'ss-vae':
        model_ss = load_model_state_dict(model_name, model_path, n_bands=109, data_path=data_path)
        model_ss.eval()  # Set the model to evaluation mode
        input_dl, _ = get_dataloaders_ssvae(data_path=config.data_path, batch_size=config.batch_size, neighborhood_size=config.patch_size, test_size=0)
        for batch in input_dl:
            x=batch.float()
            with torch.inference_mode():
                z, mean, recon = model_ss(x)
        if hasattr(recon, 'cpu'):
            recon = recon.cpu().numpy()  # shape: (effective_rows*effective_cols, B)
        recon_np = recon
        
    elif model_name == 'vae':
        model_vae = load_model_state_dict(model_name, model_path, n_bands=109, data_path=data_path)
        model_vae.eval()
        input_dl, _, _ = get_dataloaders(data_path=data_path, batch_size=32, test_size=0.0)
        recon_vectors = []
        recon_losses = []
        mse_losses = []
        sid_losses = []
        for batch in input_dl:
            x = batch[0].float()
            with torch.inference_mode():
                _, _, recon = model_vae(x)
            if len(recon)==2:
                recon = recon[0] # by mistake, if vae returns a tuple of out, E_positive. Changed its fwd method to return out only now
            recon_loss_term = spectral_angle_distance_loss(recon, x)
            mse_losse_term  = mse_loss(recon, x)
            sid_loss_term = spectral_information_divergence_loss(recon, x)
            recon_losses.append(recon_loss_term.item())
            recon_vectors.append(recon.detach().numpy())
            mse_losses.append(mse_losse_term.item())
            sid_losses.append(sid_loss_term.item())
        print(f"Average MSE Loss: {np.mean(mse_losses)}")
        print(f"Average SID Loss: {np.mean(sid_losses)}")
        print(f"Average Reconstruction Loss: {np.mean(recon_losses)}")
        recon_np = np.concatenate(recon_vectors, axis=0)
        
    elif model_name == 'ae':
        model_ae = load_model_state_dict(model_name, model_path, n_bands=109, data_path=data_path, em_path=em_path)
        input_dl, _, _ = get_dataloaders(data_path=data_path, batch_size=32, test_size=0.0)
        recon_vectors = []
        recon_losses = []
        mse_losses = []
        sid_losses = []
        for x in input_dl:
            x = x[0].float()  # type: ignore # Extract the tensor from the tuple
            with torch.inference_mode():
                x_hat, z, E_positive = model_ae(x)
            recon_loss_term = spectral_angle_distance_loss(x_hat, x)
            recon_losses.append(recon_loss_term.item())
            recon_vectors.append(x_hat.detach().numpy())
            mse_loss_term = mse_loss(x_hat, x)
            sid_loss_term = spectral_information_divergence_loss(x_hat, x)
            mse_losses.append(mse_loss_term.item())
            sid_losses.append(sid_loss_term.item())
        print(f"Average MSE Loss: {np.mean(mse_losses)}")
        print(f"Average SID Loss: {np.mean(sid_losses)}")
        print(f"Average Reconstruction Loss: {np.mean(recon_losses)}")
        recon_np = np.concatenate(recon_vectors, axis=0)
    elif model_name == 'ae2':
        model = load_model_state_dict(model_name, model_path, n_bands=156, data_path=data_path)
        input_dl, _ = samson_dataloader(data_path=data_path, batch_size=20, test_size=0.0)
        recon_vectors = []
        recon_losses = []
        # mse_losses = []
        sid_losses = []
        for x in input_dl:
            x = x[0].float()  # type: ignore # Extract the tensor from the tuple
            with torch.inference_mode():
                x_hat, z = model(x)
                x_hat = torch.nn.functional.relu(x_hat)  # ensure non-negative outputs
            recon_loss_term = spectral_angle_distance_loss(x_hat, x)
            recon_losses.append(recon_loss_term.item())
            recon_vectors.append(x_hat.detach().numpy())
            # mse_loss_term = mse_loss(x_hat, x)
            sid_loss_term = spectral_information_divergence_loss(x_hat, x)
            # mse_losses.append(mse_loss_term.item())
            sid_losses.append(sid_loss_term.item())
        # print(f"Average MSE Loss: {np.mean(mse_losses)}")
        print(f"Average SID Loss: {np.mean(sid_losses)}")
        print(f"Average Reconstruction Loss: {np.mean(recon_losses)}")
        recon_np = np.concatenate(recon_vectors, axis=0)
    return recon_np

if __name__ == "__main__":
    model_path = "/teamspace/studios/this_studio/isro-spectral-unmixing/src/models/model_state_ae_scaling0701_050635.pth"
    data_path = "/teamspace/studios/this_studio/isro-spectral-unmixing/data/den_reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz"
    em_path = "/teamspace/studios/this_studio/isro-spectral-unmixing/data/vca_ch2_iir_nci_20191208T0814159609.npy"
    config = get_config()
    abundance_vecs = extract_latent_vectors('ae', model_path=model_path, config=config, data_path=data_path, em_path=em_path) 
    recon_vectors = get_recon_spectra('ae', model_path=model_path, config=config, data_path=data_path, em_path=em_path)
    print("recon vectors: ", len(recon_vectors), recon_vectors[0].shape)
    print("abundance vecs: ", len(abundance_vecs), abundance_vecs[0].shape)