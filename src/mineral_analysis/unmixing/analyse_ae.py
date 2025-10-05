#%%
import torch
import torch.nn as nn
from typing import Tuple, List
from dimension_reduction.ss_vae.dataloaders import get_dataloaders, open_datacube
from mineral_analysis.unmixing.ae2 import AE2, spectral_angle_distance_loss
from mineral_analysis import endmember_extraction as eea
from dimension_reduction.inference_utils import extract_latent_vectors, get_recon_spectra, load_model_state_dict
from dimension_reduction.ss_vae import config
from mineral_analysis.clustering.clustering_funcs import kmeans_clustering, gmm_clustering, plot_and_eval
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np
# from dimension_reduction.latent_vectors import show_recon_image
#%%
model_path = "/teamspace/studios/this_studio/isro-spectral-unmixing/src/models/model_state_vae_unmix_1004_165825.pth"
data_path = "/teamspace/studios/this_studio/isro-spectral-unmixing/data/den_reflectance_ch2_iir_nci_20210620T2110364457_d_img_hw1 (1).npz"
em_path = None

# models_info_dict = {
#     'ae':{
#         'Image1': (
#                 "/teamspace/studios/this_studio/isro-spectral-unmixing/src/models/model_state_ae_scaling0701_050635.pth", 
#                 "/teamspace/studios/this_studio/isro-spectral-unmixing/data/den_reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz", 
#                 "/teamspace/studios/this_studio/isro-spectral-unmixing/data/vca_ch2_iir_nci_20191208T0814159609.npy",
#                 "20191208T0814159609"
#                 ),
#         'Image2': (
#             model_path, data_path, em_path,
#             "20210620T2110364457"
#             ),
#         'Image3': (model_path, data_path, em_path),
#         'Image4': (model_path, data_path, em_path),
#     },
#     'vae':{
#         'Image1': (model_path, data_path, em_path),
#         'Image2': (model_path, data_path, em_path),
#         'Image3': (model_path, data_path, em_path),
#         'Image4': (model_path, data_path, em_path),
#     },
# }

config = config.get_config()
abundance_vecs = extract_latent_vectors('vae', model_path=model_path, config=config, data_path=data_path, em_path=em_path) 
recon_vectors = get_recon_spectra('vae', model_path=model_path, config=config, data_path=data_path, em_path=em_path)

H, wavelengths = open_datacube(data_path)
H_t = H.transpose(1, 2, 0)  # Move bands to the last dimension
rows, cols, bands = H_t.shape
X_flat = H_t.reshape(rows*cols, bands)
print("Data loaded and reshaped:", X_flat.shape)

#%%
# abundance_vecs = np.concatenate(abundance_vectors, axis=0)
print("recon vectors: ", len(recon_vectors), recon_vectors[0].shape)
#Plot the abundance maps
print(abundance_vecs.shape)
amaps = abundance_vecs.reshape(H_t.shape[0], H_t.shape[1], 4)  
print(amaps.shape)
eea.plot_amaps(amaps, H_t, wavelengths, "VAE", target_wl=750)

# Plot the endmembers
# model = load_model_state_dict('ae2', model_path=model_path, n_bands=156, data_path=data_path, em_path=em_path)
# ems = model.decoder_output_layer.weight.detach().cpu().numpy() # type: ignore
# ems = ems.T
# print("Endmembers shape from decoder weights:", ems.shape)
# eea.plot_endmembers(ems, None, title="Endmembers from AE Decoder Weights")
# %%
# # Clustering
kmeans_labels, score = kmeans_clustering(X_flat, n_clusters=4, rows=rows, cols=cols)
print("KMeans clustering of original image completed.", kmeans_labels.shape)
print("Silhouette Score:", score)
_ = plot_and_eval(H, 4, kmeans_labels, "KMeans", img_name="20214457")

kmeans_labels, score = kmeans_clustering(abundance_vecs, n_clusters=4, rows=rows, cols=cols)
plot_and_eval(H,4, kmeans_labels, "KMeans VAE", img_name="20214457", img_type='latent', model_name='vae')
print("KMeans clustering on VAE abundance maps completed.", kmeans_labels.shape)
print("Silhouette Score for VAE abundance maps:", score)


#%%