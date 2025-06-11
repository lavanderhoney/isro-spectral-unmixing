"""
Applies mineral endmember extraction on the latent vectors of a hyperspectral cube.
Show the reconstructed image, computes the SAM, and plots the original and reconstructed spectra.
"""
import numpy as np
from typing import Literal
from matplotlib import pyplot as plt
from mineral_analysis.endmember_extraction import extract_endmembers
from dimension_reduction.inference_utils import get_recon_spectra

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

    # get the reconstructed spectra/image
    recon_np = get_recon_spectra(
        model_name=model_name,
        model_path=model_path,
        input_data=input_data,
    )

    # Reshape input for plotting
    cube = input_data.reshape(rows, cols, -1)
    if model_name == 'ss-vae':
        # Center-crop original cube to match latent output size
        effective_rows, effective_cols = 997, 246
        start_r = (rows - effective_rows) // 2
        start_c = (cols - effective_cols) // 2
        cube = cube[start_r:start_r + effective_rows, start_c:start_c + effective_cols, :]
    else:
        effective_rows, effective_cols = rows, cols

    orig_band = cube[:, :, band_index]

    # Extract reconstructed band and normalize both for display
    recon_band = recon_np.mean(axis=1).reshape(effective_rows, effective_cols)
    orig_norm = (orig_band - orig_band.min()) / (orig_band.max() - orig_band.min() + 1e-6)
    recon_norm = (recon_band - recon_band.min()) / (recon_band.max() - recon_band.min() + 1e-6)

    # Plot side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    im1 = ax1.imshow(orig_norm, cmap='gray', vmin=0, vmax=1)
    ax1.set_title("Original Band 30")
    ax1.axis('off')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    print("recon range: ", recon_band.min(), recon_band.max())
    im2 = ax2.imshow(recon_band, cmap='gray', vmin=0, vmax=1)
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
        return cos_theta

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
    show_recon_image('ss-vae', ss_vae_path, input_data, rows, cols, n_samples=3)
    # show_recon_image('vae', vae_path, input_data,rows, cols, n_samples=3)
# %%