import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from skimage.restoration import denoise_tv_chambolle
from typing import Tuple
from argparse import Namespace
import os
"""
Data downloaded using this script: https://www.kaggle.com/code/milapp180/fc-vae-iirs-dim-reduction
It is noisy reflectance data, of shape 109 x 1001 x 250, with: B x H x W
"""
# %%
# refl_cube_path = '/teamspace/studios/this_studio/isro-spectral-unmixing/data/reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz'

# unloaded = np.load(refl_cube_path)
# noisy_refl_data = unloaded['refl_data']
# wavelengths = unloaded['wavelengths']

# #---- denoise the reflectance data. ----# # not storing denoised data, so that we can change the denoising method here later
# refl_data: np.ndarray = denoise_tv_chambolle(noisy_refl_data, max_num_iter=50, weight=20)
# refl_cube_path = '/teamspace/studios/this_studio/isro-spectral-unmixing/data/den_reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz' #the denoised image
# unloaded = np.load(refl_cube_path)
# refl_data = unloaded['den_refl_data']
# wavelengths = unloaded['wavelengths']
# print("Denoised image extracted!")

# spectral_bands = refl_data.shape[0]
# # %% 
# #---- normalize and create batched sequence data for the LSTM ----#
# neighborhood_size = 5 # s So seq lenght = sxs


def extract_patches(data_cube: torch.Tensor, s: int = 5) -> torch.Tensor:
    B, H, W = data_cube.shape
    
    #add batch dimension for torch.nn.functional.unfold
    data_cube = data_cube.unsqueeze(0) # (1, B, H, W), consistent with (batch, channel, *)
    
    #use PyTorch's unfold to extract patches, rather than manually iterating, which is dumb and slow
    patches = F.unfold(data_cube, kernel_size=(s, s), stride=1).squeeze(0) # (B *(s*s), N)
    N = patches.shape[-1] # number of patches
    
    # transpose and reshape to get patches in the shape (N, B, s, s)
    patches = patches.permute(0, 1).reshape(N, B, s, s)
    return patches
# %%
def split_norm_patches(patches: torch.Tensor,  test_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits the patches into training and test sets, normalizes them, and returns the normalized patches.
    """
    # Split into train and test sets
    if test_size ==0: # used for inference, so no split
        patches_train, patches_test = patches.numpy(), patches.numpy()
        print(patches_train.shape, patches_test.shape)
    else:
        patches_train, patches_test = train_test_split(patches.numpy(), test_size=0.1, random_state=42)
    # Normalize the patches
    mean = patches_train.mean(axis=(0, 2, 3), keepdims=True)  # mean across (B, s, s)
    std = patches_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8  # std across (B, s, s)
    patches_train_n = (patches_train - mean) / std
    patches_test_n = (patches_test - mean) / std
    
    return patches_train_n, patches_test_n

#%%
class SSVAEDataset(Dataset):
    def __init__(self, patches: np.ndarray, patch_size: int =5):
        self.patches = torch.Tensor(patches)
        self.s = patch_size
        self.B = self.patches.shape[1]
        # self.patches shape = (N, B, s, s) where N is number of patches, B is number of bands, s is patch size

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, idx):
        patch = self.patches[idx]                 # (B, s, s)

        # 3) Permute so that spatial dims come first, spectral last:
        #    (B, s, s) → (s, s, B)
        patch = patch.permute(1, 2, 0)             # (s, s, B)
        
        return patch  # shape = (s, s, B)

# main function used when importing this module
def get_dataloaders(data_path:str, batch_size: int = 32, neighborhood_size: int=5, test_size: float=0.1) -> Tuple[DataLoader, DataLoader]:
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path {data_path} does not exist. Please check the path.")
    unloaded = np.load(data_path)
    if 'den_refl_data' not in unloaded or 'wavelengths' not in unloaded:
        raise KeyError("The loaded data does not contain 'den_refl_data' or 'wavelengths'. Please check the file format. \
                       Ensure the file is a valid .npz file with the expected keys.")
    refl_data = unloaded['den_refl_data']
    wavelengths = unloaded['wavelengths']
    patches = extract_patches(torch.Tensor(refl_data), neighborhood_size)
    print("patches extracted:", patches.shape)

    patches_train_n, patches_test_n = split_norm_patches(patches, test_size)
    patched_data_train = SSVAEDataset(patches_train_n, neighborhood_size)
    patched_data_test = SSVAEDataset(patches_test_n, neighborhood_size)

    train_loader = DataLoader(patched_data_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(patched_data_test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# %%
if __name__ == "__main__":
    from src.dimension_reduction.ss_vae.config import get_config
    config = get_config()
    config.data_path = '/teamspace/studios/this_studio/isro-spectral-unmixing/data/den_reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz'  # Set the data path to the denoised reflectance data
    train_loader, test_loader = get_dataloaders(config.data_path, batch_size=32)
    for batch in train_loader:
        print("Batch shape:", batch.shape)  # (batch_size, s, s, B)
        break
    # np.savez_compressed('den_reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz', den_refl_data=refl_data, wavelengths=wavelengths)
    # print("Denoised reflectance data saved to 'den_reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz'")


# %%
"""
this should be the general dataset loader, except make the dl return s,s,B
then, for each of three different models, they will transform this data into their shape, during the forward pass
so, need to write utility functions that do these transformations
"""