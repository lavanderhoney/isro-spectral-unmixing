import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from skimage.restoration import denoise_tv_chambolle
from typing import Tuple, Literal
from argparse import Namespace
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.dimension_reduction.ss_vae.config import get_config
import os

"""
Data downloaded using this script: https://www.kaggle.com/code/milapp180/fc-vae-iirs-dim-reduction
It is noisy reflectance data, of shape 109 x 1001 x 250, with: B x H x W

Apparently, normalizing the data is messing with the spectra and the overall distribution of reflectance values, so stop normalizing the data.
But min-max is still fine
"""

def open_datacube(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Opens the data cube from the given path and returns the reflectance data and wavelengths.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path {data_path} does not exist. Please check the path.")
    
    unloaded = np.load(data_path)
    if 'den_refl_data' not in unloaded or 'wavelengths' not in unloaded:
        raise KeyError("The loaded data does not contain 'den_refl_data' or 'wavelengths'. Please check the file format. \
                       Ensure the file is a valid .npz file with the expected keys.")
    
    refl_data = unloaded['den_refl_data']
    wavelengths = unloaded['wavelengths']
    
    return refl_data, wavelengths

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
def split_norm_patches(patches: torch.Tensor,  test_size: float, scaling:str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits the patches into training and test sets, normalizes them, and returns the normalized patches.
    """
    # Split into train and test sets
    if test_size ==0: # used for inference, so no split
        patches_train, patches_test = patches.numpy(), patches.numpy()
        print(patches_train.shape, patches_test.shape)
    else:
        patches_train, patches_test = train_test_split(patches.numpy(), test_size=0.1, random_state=42)
        
    if scaling == 'minmax':
        patches_train = (patches_train - patches_train.min(axis=(0, 2, 3), keepdims=True)) / (patches_train.max(axis=(0, 2, 3), keepdims=True) - patches_train.min(axis=(0, 2, 3), keepdims=True) + 1e-8)
        patches_test = (patches_test - patches_train.min(axis=(0, 2, 3), keepdims=True)) / (patches_train.max(axis=(0, 2, 3), keepdims=True) - patches_train.min(axis=(0, 2, 3), keepdims=True) + 1e-8)
    elif scaling == 'standard':
        # Normalize the patches
        mean = patches_train.mean(axis=(0, 2, 3), keepdims=True)  # mean across (B, s, s)
        std = patches_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8  # std across (B, s, s)
        patches_train_n = (patches_train - mean) / std
        patches_test_n = (patches_test - mean) / std
    else:
        patches_train_n = patches_train
        patches_test_n = patches_test
    
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
def get_dataloaders_ssvae(data_path:str, batch_size: int = 32, neighborhood_size: int=5, test_size: float=0.1, scaling: Literal['minmax', 'standard', 'none'] = 'none') -> Tuple[DataLoader, DataLoader]:

    refl_data, wavelengths = open_datacube(data_path)
    patches = extract_patches(torch.Tensor(refl_data), neighborhood_size)
    print("patches extracted:", patches.shape)

    patches_train_n, patches_test_n = split_norm_patches(patches, test_size, scaling)
    patched_data_train = SSVAEDataset(patches_train_n, neighborhood_size)
    patched_data_test = SSVAEDataset(patches_test_n, neighborhood_size)

    train_loader = DataLoader(patched_data_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(patched_data_test, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader

def get_dataloaders(data_path: str, 
                     batch_size: int = 32, 
                     test_size: float = 0.1, 
                     scaling: Literal['minmax', 'standard', 'none'] = 'none') -> Tuple[DataLoader, DataLoader, np.ndarray]:
    """
    Return dataloaders and wavelengths for the other models. Just wraps numpy dataset in a PyTorch Dataset and DataLoader.
    """
    refl_data, wavelengths = open_datacube(data_path)
    H_t = np.moveaxis(refl_data, 0, 2)

    rows, cols, bands = H_t.shape
    X_flat = H_t.reshape(rows*cols, bands)

    X_flat_train, X_flat_test = train_test_split(X_flat, test_size=test_size, shuffle=True)
    if scaling == 'minmax':
        scaler = MinMaxScaler()
    elif scaling == 'standard':
        scaler = StandardScaler()
    else:
        scaler = None

    if scaler:
        X_flat_train_norm = scaler.fit_transform(X_flat_train) if scaler else X_flat_train
        X_flat_test_norm = scaler.transform(X_flat_test) if scaler else X_flat_test
    else:
        X_flat_train_norm = X_flat_train
        X_flat_test_norm = X_flat_test

    train_data = TensorDataset(torch.tensor(X_flat_train_norm))
    test_data = TensorDataset(torch.tensor(X_flat_test_norm))
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    return train_dl, test_dl, wavelengths


# %%
if __name__ == "__main__":
    
    config = get_config()
    config.data_path = '/teamspace/studios/this_studio/isro-spectral-unmixing/data/den_reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz'  # Set the data path to the denoised reflectance data
    train_loader, test_loader = get_dataloaders_ssvae(config.data_path, batch_size=32)
    for batch in train_loader:
        print("Batch shape:", batch.shape)  # (batch_size, s, s, B)
        break
    
    train_dl, test_dl, _ = get_dataloaders(config.data_path, batch_size=32)
    for batch in train_dl:
        print("Batch shape for get_dataloaders:", batch[0].shape)  # (batch_size, B)
        break
    # np.savez_compressed('den_reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz', den_refl_data=refl_data, wavelengths=wavelengths)
    # print("Denoised reflectance data saved to 'den_reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz'")


