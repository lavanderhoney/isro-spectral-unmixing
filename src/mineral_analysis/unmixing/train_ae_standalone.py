"""
Self-contained training script for the Linear Mixture Model (LMM) based Autoencoder.
This file contains all necessary code to train the AE model without external dependencies.

Usage:
    python train_ae_standalone.py
"""

import numpy as np
import scipy
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from time import sleep
from tqdm import tqdm
from typing import Tuple, Literal, Union, Optional
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
from datetime import datetime

# ==================== Configuration ====================
class Config:
    def __init__(self):
        self.data_path = '/teamspace/studios/this_studio/isro-spectral-unmixing/data/den_reflectance_ch2_iir_nci_20210620T2110364457_d_img_hw1 (1).npz'
        self.test_size = 0.1
        self.batch_size = 32
        self.hidden_dim = 64
        self.latent_dim = 4
        self.epochs = 30
        self.lr = 1e-4
        self.update_interval = 1000
        self.early_stop = 5
        self.scheduler_patience = 3
        self.scaling = 0.8  # Softmax scaling parameter
        self.constraint = 'softmax'  # 'softmax' or 'man'
        self.n_endmembers = 4

def get_config():
    return Config()

# ==================== Metrics Logger ====================
class MetricsLogger:
    def __init__(self) -> None:
        """
        Logs and tracks training and validation metrics during the training process.
        """
        self.reset_epoch()
        self.history = {
            'train': {'total': [], 'recon': [], 'kl': [], 'homology': [], 'mv': [], 'sid': []},
            'val': {'total': [], 'recon': [], 'kl': [], 'homology': [], 'mv': [], 'sid': []}
        }
        self.regularization_losses = [False, False, False, False]  # kl, homology, mv, sid
    
    def reset_epoch(self) -> None:
        """Resets the metrics for a new epoch."""
        self.epoch_metrics = {
            'train': {'total': 0.0, 'recon': 0.0, 'kl': 0.0, 'homology': 0.0, 'mv':0.0, 'sid':0.0, 'count': 0},
            'val': {'total': 0.0, 'recon': 0.0, 'kl': 0.0, 'homology': 0.0, 'mv':0.0, 'sid':0.0, 'count': 0}
        }
    
    def update(self, 
               phase: Literal['train', 'val'], 
               total: float, 
               recon: float, 
               kl: Optional[float]=None, 
               homology: Optional[float]=None, 
               mv: Optional[float]=None, 
               sid: Optional[float]=None
               ) -> None:
        """Updates the metrics for the current epoch."""
        metrics = self.epoch_metrics[phase]
        metrics['total'] += total
        metrics['recon'] += recon
        if kl:
            metrics['kl'] += kl
            self.regularization_losses[0] = True
        if homology:
            metrics['homology'] += homology
            self.regularization_losses[1] = True
        if mv:
            metrics['mv'] += mv
            self.regularization_losses[2] = True
        if sid:
            metrics['sid'] += sid
            self.regularization_losses[3] = True
        metrics['count'] += 1
    
    def finalize_epoch(self, phase: Literal['train', 'val']) -> float:
        """Finalizes the metrics for the current epoch and appends them to history."""
        metrics = self.epoch_metrics[phase]
        count = metrics['count']
        self.history[phase]['total'].append(metrics['total'] / count)
        self.history[phase]['recon'].append(metrics['recon'] / count)
        self.history[phase]['kl'].append(metrics['kl'] / count)
        self.history[phase]['homology'].append(metrics['homology'] / count)
        self.history[phase]['mv'].append(metrics['mv'] / count)
        self.history[phase]['sid'].append(metrics['sid'] / count)
        return self.history[phase]['total'][-1]

# ==================== Data Loading ====================
def open_datacube(data_path: str) -> Tuple[np.ndarray, np.ndarray|None]:
    """Opens the data cube from the given path and returns reflectance data and wavelengths."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path {data_path} does not exist. Please check the path.")
    
    unloaded = np.load(data_path)
    
    if 'den_refl_data' not in unloaded:
        raise KeyError("The loaded data does not contain 'den_refl_data'. Please check the file format.")
    
    refl_data = unloaded['den_refl_data']
    if 'wavelengths' in unloaded:
        wavelengths = unloaded['wavelengths']
    else:
        wave = np.load('/teamspace/studios/this_studio/isro-spectral-unmixing/data/m3_wavelengths_Copy of m3g20090729t104424.npz', allow_pickle=True)
        wavelengths = wave.get('wavelengths', None)

    if refl_data.ndim != 3 or (refl_data.shape[0] != 109 and refl_data.shape[0]!=85):
        raise ValueError(f"Expected reflectance data to have shape (B, H, W), but got {refl_data.shape}. Please check the data format.")
    return refl_data, wavelengths

def get_dataloaders(data_path: str, 
                     batch_size: int = 32, 
                     test_size: float = 0.1, 
                     scaling: Literal['minmax', 'standard', 'none'] = 'none') -> Tuple[DataLoader, DataLoader, np.ndarray|None]:
    """Return dataloaders and wavelengths for the models."""
    refl_data, wavelengths = open_datacube(data_path)
    
    if refl_data.shape[2] != 109:  # check if bands are in last dimension
        H_t = np.moveaxis(refl_data, 0, 2)
    else:
        raise ValueError("The input data cube should be in shape (B, H, W)")
    
    rows, cols, bands = H_t.shape
    X_flat = H_t.reshape(rows*cols, bands)

    if test_size == 0.0:  # used for inference, so no split
        X_flat_train, X_flat_test = X_flat, X_flat
    else:
        X_flat_train, X_flat_test = train_test_split(X_flat, test_size=test_size, shuffle=False)
    
    if scaling == 'minmax':
        scaler = MinMaxScaler()
    elif scaling == 'standard':
        scaler = StandardScaler()
    else:
        scaler = None

    if scaler:
        X_flat_train_norm = scaler.fit_transform(X_flat_train)
        X_flat_test_norm = scaler.transform(X_flat_test)
    else:
        X_flat_train_norm = X_flat_train
        X_flat_test_norm = X_flat_test

    train_data = TensorDataset(torch.tensor(X_flat_train_norm))
    test_data = TensorDataset(torch.tensor(X_flat_test_norm))
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_dl, test_dl, wavelengths

# ==================== Endmember Extraction (VCA) ====================
# Monkey patch for scipy and numpy compatibility with pysptools
class _FakeFLinalg:
    @staticmethod
    def sdet_c(mat):
        return (np.linalg.det(mat), 0)

if not hasattr(scipy.linalg, '_flinalg'):
    scipy.linalg._flinalg = _FakeFLinalg()

if not hasattr(np, 'int'):
    np.int = int

def extract_endmembers_vca(H_t: np.ndarray, 
                           wavelengths: np.ndarray, 
                           n_endmembers: int = 5) -> np.ndarray:
    """
    Extract endmembers from a hyperspectral cube using VCA algorithm.
    
    Args:
        H_t: (rows, cols, bands) hyperspectral data cube
        wavelengths: (bands,) array of wavelength values
        n_endmembers: number of endmembers to extract
    
    Returns:
        endmembers: (n_endmembers, bands) array of extracted endmember spectra
    """
    try:
        import ramanspy as rp
        image = rp.SpectralImage(H_t, spectral_axis=wavelengths)
        vca = rp.analysis.unmix.VCA(n_endmembers=n_endmembers, abundance_method='fcls')
        abundance_maps_vca, em_vca = vca.apply(image)
        em_vca_np = np.array(em_vca)
        return em_vca_np
    except ImportError:
        print("Warning: ramanspy not available. Using random endmembers initialization.")
        # Fallback: Use random samples from the dataset
        rows, cols, bands = H_t.shape
        X_flat = H_t.reshape(rows*cols, bands)
        indices = np.random.choice(X_flat.shape[0], n_endmembers, replace=False)
        return X_flat[indices]

# ==================== AE Model ====================
class AE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, em_spectra, 
                 constraint: Literal['softmax', 'man'] = 'softmax', scaling: float = 0.8) -> None:
        super().__init__()
        # Encoder layers
        self.enc_linear1 = nn.Linear(input_dim, hidden_dim)
        self.enc_bn1 = nn.BatchNorm1d(hidden_dim)
        self.enc_act1 = nn.LeakyReLU()

        self.enc_linear2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.enc_bn2 = nn.BatchNorm1d(hidden_dim//2)
        self.enc_act2 = nn.LeakyReLU()

        self.enc_linear3 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.enc_bn3 = nn.BatchNorm1d(hidden_dim//4)
        self.enc_act3 = nn.LeakyReLU()
        
        self.encoder_output_layer = nn.Linear(hidden_dim//4, latent_dim)
        
        # Endmember Matrix E
        E_tensor = torch.from_numpy(em_spectra).float()
        self.E = nn.Parameter(E_tensor, requires_grad=False)  # M x N matrix
        self.constraint = constraint
        self.beta = scaling

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc_linear1(x)
        x = self.enc_bn1(x)
        x = self.enc_act1(x)
        
        x = self.enc_linear2(x)
        x = self.enc_bn2(x)
        x = self.enc_act2(x)
      
        x = self.enc_linear3(x)
        x = self.enc_bn3(x)
        x = self.enc_act3(x)

        z = self.encoder_output_layer(x)
        
        # ANC and ASC constraints
        if self.constraint == 'softmax':
            z = F.softmax(z/self.beta, dim=1)
        elif self.constraint == 'man':
            z = F.relu(z)  # ANC
            z = z / (torch.sum(z, dim=1, keepdim=True) + 1e-6)  # ASC
        else:
            raise ValueError("Invalid constraint type. Use 'softmax' or 'man'.")
        return z

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        E_positive = F.relu(self.E)
        assert not torch.isnan(E_positive).any(), "NaN in E_positive"
        assert not torch.isnan(z).any(), "NaN in z"
        x_hat = torch.matmul(z, E_positive)
        
        if torch.isnan(x_hat).any() or torch.isinf(x_hat).any():
            print("DEBUG: NaN/Inf in x_hat")
            print("z stats:", torch.min(z).item(), torch.max(z).item(), torch.mean(z).item())
            print("E_positive stats:", torch.min(E_positive).item(), torch.max(E_positive).item(), torch.mean(E_positive).item())
            raise RuntimeError("NaN/Inf detected in x_hat")
        return x_hat, E_positive

    def forward(self, x):
        z = self.encode(x)
        x_hat, E_positive = self.decode(z)
        return x_hat, z, E_positive

# ==================== Loss Functions ====================
def spectral_angle_distance_loss(x_hat: torch.Tensor, x: torch.Tensor, eps:float = 1e-8) -> torch.Tensor:
    """Compute Spectral Angle Distance (SAD) loss."""
    cos_theta = F.cosine_similarity(x_hat, x, dim=1, eps=eps)
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-4, 1.0 - 1e-4)
    angle = torch.acos(cos_theta)
    return torch.mean(angle)

def spectral_information_divergence_loss(x_hat: torch.Tensor, x: torch.Tensor, eps:float = 1e-8) -> torch.Tensor:
    """Compute Spectral Information Divergence (SID) loss."""
    x_hat, x = x_hat + eps, x + eps
    p = x / torch.sum(x, dim=-1, keepdim=True)
    q = x_hat / torch.sum(x_hat, dim=-1, keepdim=True)

    kl_pq = torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)
    kl_qp = torch.sum(q * (torch.log(q) - torch.log(p)), dim=-1)
    sid = kl_pq + kl_qp
    return torch.mean(sid)
    
def total_variation(E: torch.Tensor) -> torch.Tensor:
    """Compute Total Variation function TV_r(E) for the endmember matrix E."""
    M, N = E.shape
    r = M
    ones_M = torch.ones(M, 1, device=E.device)
    I_r = torch.eye(M, device=E.device, dtype=E.dtype)
    P = I_r - 1/r * (ones_M @ ones_M.T)
    TV = torch.norm(torch.matmul(E.T, P), p='fro') ** 2
    return TV

def entropy_loss(z: torch.Tensor) -> torch.Tensor:
    """Compute entropy loss for the latent abundances z."""
    entropy = -torch.sum(z * torch.log(z + 1e-8), dim=1)
    return torch.mean(entropy)

# ==================== Training Function ====================
def main():
    """Main function to train the LMM based Autoencoder (AE) on reflectance data."""
    config = get_config()
    
    # Load and preprocess the reflectance data
    print(f"Loading data from: {config.data_path}")
    train_dl, test_dl, wavelengths = get_dataloaders(config.data_path, batch_size=config.batch_size, test_size=config.test_size)
    H, wavelengths = open_datacube(config.data_path)
    H_t = H.transpose(1, 2, 0)  # (H, W, bands)
    
    # Get number of spectral bands
    first_batch = next(iter(train_dl))
    n_bands = first_batch[0].shape[1]
    
    metrics = MetricsLogger()
    
    # Extract endmembers using VCA
    print("Extracting endmembers using VCA...")
    ems = extract_endmembers_vca(H_t, wavelengths, n_endmembers=config.n_endmembers)
    print(f"Endmembers shape: {ems.shape}")
    print("VCA Done")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = AE(
        input_dim=n_bands,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        em_spectra=ems,
        constraint=config.constraint,
        scaling=config.scaling
    ).to(device)
    
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=config.scheduler_patience, threshold=0.01)
    
    patience_cntr = 0
    best_model_state = None
    best_test_loss = float('inf')
    
    print("The Training Begins!")
    sleep(0.5)

    w_recon, w_mv, w_sid, w_ent = 1, 1, 1, 0  # Weights for the losses
    
    for epoch in range(config.epochs):
        metrics.reset_epoch()

        # TRAINING
        model.train()
        train_pbar = tqdm(train_dl, total=len(train_dl), desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        for i, x in enumerate(train_pbar):
            x = x[0].float().to(device)
            
            optim.zero_grad()
           
            x_hat, z, E_positive = model(x)

            recon_loss_term = spectral_angle_distance_loss(x_hat, x)
            mv_loss_term = total_variation(E_positive)
            sid_loss_term = spectral_information_divergence_loss(x_hat, x)
            entropy_loss_term = entropy_loss(z)
            loss = w_recon*recon_loss_term + w_mv*mv_loss_term + w_sid*sid_loss_term - w_ent*entropy_loss_term

            loss.backward()
            optim.step()

            metrics.update(phase='train', total=loss.item(), recon=recon_loss_term.item(), 
                         mv=mv_loss_term.item(), sid=sid_loss_term.item(), homology=entropy_loss_term.item())

            if i % config.update_interval == 0:
                train_pbar.set_postfix({
                    "loss": "{:.5f}".format(loss.item()),
                    "reconstruction": "{:.4f}".format(recon_loss_term.item()),
                    "total_variation": "{:.4f}".format(mv_loss_term.item()),
                    "sid_loss": "{:.4f}".format(sid_loss_term.item()),
                    "entropy": "{:.4f}".format(entropy_loss_term.item())
                })
                if math.isnan(loss.item()):
                    print(E_positive)
                    print(z)
                    print(x_hat)
                    for name, param in model.named_parameters():
                        if torch.isnan(param).any():
                            print(f"Parameter {name} has NaN values.")
                    raise ValueError("Loss went to nan.")
        
        avg_train_loss = metrics.finalize_epoch('train')
        
        # EVALUATION
        model.eval()
        test_pbar = tqdm(test_dl, total=len(test_dl), desc=f"Epoch {epoch+1}/{config.epochs} [Test]")
        with torch.no_grad():
            for x in test_pbar:
                x = x[0].float().to(device)

                x_hat, z, E_positive = model(x)
                recon_loss_term = spectral_angle_distance_loss(x_hat, x)
                mv_loss_term = total_variation(E_positive)
                sid_loss_term = spectral_information_divergence_loss(x_hat, x)
                entropy_loss_term = entropy_loss(z)
                loss = w_recon*recon_loss_term + w_mv*mv_loss_term + w_sid*sid_loss_term + w_ent*entropy_loss_term

                metrics.update(phase='val', total=loss.item(), recon=recon_loss_term.item(), 
                             mv=mv_loss_term.item(), sid=sid_loss_term.item(), homology=entropy_loss_term.item())
                test_pbar.set_postfix({
                    "loss": "{:.5f}".format(loss.item()),
                    "reconstruction": "{:.4f}".format(recon_loss_term.item()),
                    "total_variation": "{:.4f}".format(mv_loss_term.item()),
                    "sid_loss": "{:.4f}".format(sid_loss_term.item()),
                    "entropy": "{:.4f}".format(entropy_loss_term.item())
                })

        avg_test_loss = metrics.finalize_epoch('val')
        scheduler.step(avg_test_loss)

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            print("New best model found.")
            best_model_state = model.state_dict()
            patience_cntr = 0
        else:
            patience_cntr += 1

        if patience_cntr >= config.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return model, best_model_state, metrics

# ==================== Main Entry Point ====================
if __name__ == "__main__":
    model, best_model_state, metrics = main()
   
    # Save the best model state
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    state = {
        'model_state': model.state_dict(),
        'metrics': metrics,
        'config': get_config(),
        'timestamp': timestamp,
        'scaling': model.beta
    }
    torch.save(state, f'models/model_state_ae_scaling{timestamp}.pth')
    print(f"Best model saved to 'models/model_state_ae_scaling{timestamp}.pth'")
    print("Training complete!")
