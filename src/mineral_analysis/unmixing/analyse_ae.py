#%%
import torch
import torch.nn as nn
from typing import Tuple, List
from dimension_reduction.ss_vae.dataloaders import get_dataloaders, open_datacube
from mineral_analysis.unmixing.ae2 import AE2, spectral_angle_distance_loss, total_variation
from mineral_analysis import endmember_extraction as eea
from dimension_reduction.inference_utils import extract_latent_vectors, get_recon_spectra, load_model_state_dict
from dimension_reduction.ss_vae import config
from matplotlib import pyplot as plt
import numpy as np
# from dimension_reduction.latent_vectors import show_recon_image
#%%
model_path = "/teamspace/studios/this_studio/isro-spectral-unmixing/src/models/model_ae20922_095236.pth"
data_path = "/teamspace/studios/this_studio/isro-spectral-unmixing/data/den_reflectance_ch2_iir_nci_20191208T1407123802_d_img_d18.npz"
em_path = "/teamspace/studios/this_studio/isro-spectral-unmixing/data/vca_ch2_iir_nci_20191208T1407123802_d_img_d18.npy"
config = config.get_config()
abundance_vecs = extract_latent_vectors('ae', model_path=model_path, config=config, data_path=data_path, em_path=em_path) 
recon_vectors = get_recon_spectra('ae', model_path=model_path, config=config, data_path=data_path, em_path=em_path)
# state = torch.load(model_path, map_location='cpu', weights_only=False)
# raw_state_dict = state['model_state'] if 'model_state' in state else state
# Remove '_orig_mod.' prefix from all keys
# cleaned_state_dict = {
#     k.replace("_orig_mod.", ""): v
#     for k, v in raw_state_dict.items()
# }


H, wavelengths = open_datacube(data_path)
H_t = H.transpose(1, 2, 0)  # Move bands to the last dimension

# input_dl, _, _ = get_dataloaders(data_path, batch_size=32, test_size=0.0)
# first_batch = next(iter(input_dl))
# print(first_batch, type(first_batch))
# n_bands = first_batch[0].shape[1]  # type: ignore # Number of spectral bands
# # ems, _ = eea.extract_endmembers(H_t, wavelengths, algorithm='vca', n_endmembers=4, show_endmembers=False, show_amaps=False)
# model_ae = AE2(
#         input_dim=n_bands,  # Number of spectral bands
#         hidden_dim=state['config'].hidden_dim,
#         latent_dim=4,
#         scaling=0.8)
# model_ae.load_state_dict(cleaned_state_dict)
# model_ae.eval()
# print(input_dl.batch_size, len(input_dl))

# abundance_vectors = []
# recon_vectors = []
# recon_losses = []
# for x in input_dl:
#     x = x[0].float()  # type: ignore # Extract the tensor from the tuple
#     with torch.inference_mode():
#         x_hat, z = model_ae(x)
#     recon_loss_term = spectral_angle_distance_loss(x_hat, x)
#     mv_loss_term = total_variation(model_ae.decoder_output_layer.weight)
#     loss = recon_loss_term + state['config'].gamma * mv_loss_term
#     recon_losses.append(recon_loss_term.item())
#     abundance_vectors.append(z.detach().numpy())
#     recon_vectors.append(x_hat.detach().numpy())
# print("Avg recon loss: ", np.mean(recon_losses))
#%%
# abundance_vecs = np.concatenate(abundance_vectors, axis=0)
print("recon vectors: ", len(recon_vectors), recon_vectors[0].shape)
#Plot the abundance maps
print(abundance_vecs.shape)
amaps = abundance_vecs.reshape(H_t.shape[0], H_t.shape[1], 4)  # Assuming 4 endmembers
print(amaps.shape)
eea.plot_amaps(amaps, H_t, wavelengths, "AE", target_wl=750)

#Plot the endmember spectra
# em_spectra = model_ae.decoder_output_layer.weight.detach().numpy()
# print("Endmember spectra shape: ", em_spectra.shape)
# em_spectra = np.moveaxis(em_spectra, 0, 1)  # Shape (bands, n_endmembers)
# eea.plot_endmembers(em_spectra, wavelengths, title="AE Endmember Spectra", algorithm_name="AE")
# %%