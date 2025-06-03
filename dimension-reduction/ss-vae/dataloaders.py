import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from skimage.restoration import denoise_tv_chambolle

"""
Data downloaded using this script: https://www.kaggle.com/code/milapp180/fc-vae-iirs-dim-reduction
It is noisy reflectance data, of shape 109 x 1001 x 250, with: B x H x W
"""
# %%
refl_cube_path = '/teamspace/studios/this_studio/isro-spectral-unmixing/data/reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz'

unloaded = np.load(refl_cube_path)
noisy_refl_data = unloaded['refl_data']
wavelengths = unloaded['wavelengths']

#---- denoise the reflectance data. ----# # not storing denoised data, so that we can change the denoising method here later
refl_data = denoise_tv_chambolle(noisy_refl_data, max_num_iter=50, weight=20)
print("Image extracted and denoised.")


# %% 
#---- normalize and create batched sequence data for the LSTM ----#
s = 5 # neighorhood size. So seq lenght = sxs


def extract_sequences(data_cube: torch.Tensor, s: int = 5) -> torch.Tensor:
    B, H, W = data_cube.shape
    
    #add batch dimension for torch.nn.functional.unfold
    data_cube = data_cube.unsqueeze(0) # (1, B, H, W), consistent with (batch, channel, *)
    
    #use PyTorch's unfold to extract patches, rather than manually iterating, which is dumb and slow
    sequences = F.unfold(data_cube, kernel_size=(s, s), stride=1).squeeze(0) # (B *(s*s), N)
    N = sequences.shape[-1] # number of patches
    
    # transpose and reshape to get patches in the shape (N, B, s, s)
    sequences = sequences.permute(0, 1).reshape(N, B, s, s)
    return sequences

sequences = extract_sequences(torch.Tensor(refl_data), s)
print("Sequences extracted:", sequences.shape)
# %%

sequences_train, sequences_test = train_test_split(sequences.numpy(), test_size=0.2, random_state=42)

mean = sequences_train.mean(axis=(0, 2, 3), keepdims=True) #these are numpy's methods
std = sequences_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8

sequences_train_n = (sequences_train - mean) / std
sequences_test_n = (sequences_test - mean) / std

#%%
class SequentialDataset(Dataset):
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

        # 4) Flatten the s×s spatial grid into sequence length s^2:
        #    → (s*s, B)
        # this is compatible with the LSTM input shape
        seq = patch.reshape(self.s * self.s, self.B)

        return seq  # shape = (s^2, B)

sequences_data_train = SequentialDataset(sequences_train_n, s)
sequences_data_test = SequentialDataset(sequences_test_n, s)

def get_dataloaders(batch_size: int = 32):
    train_loader = DataLoader(sequences_data_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(sequences_data_test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders(batch_size=32)
    for batch in train_loader:
        print("Batch shape:", batch.shape)  # Should be (batch_size, B, s, s)
        break
# %%
