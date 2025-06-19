#%%
"""
Applies the classical algorithms: N-FINDR, VCA, FIPPI and ATGP, and optionally compute and plot abundance maps.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pysptools
import gc
import ramanspy as rp
from typing import Literal, Union, Tuple
from pysptools import eea
from ramanspy.analysis.unmix import VCA
from pysptools import abundance_maps as amp


# ------ Monkey patch for scipy and numpy for compatibility with old versions used by pysptools ------

# Create a dummy _flinalg namespace on scipy.linalg
class _FakeFLinalg:
    @staticmethod
    def sdet_c(mat):
        # return (determinant, info) just like the old Fortran wrapper did
        return (np.linalg.det(mat), 0)

if not hasattr(scipy.linalg, '_flinalg'):
    scipy.linalg._flinalg = _FakeFLinalg() # type: ignore

# Patch for deprecated `np.int`
if not hasattr(np, 'int'):
    np.int = int #type: ignore
# --------------------------------------------------------------------------------------------

def plot_endmembers(endmembers, wavelengths, title="Extracted Endmember Spectra", algorithm_name=""):
    """
    - endmembers: (q x bands) numpy array of extracted endmember spectra
    - wavelengths: (bands,) array of wavelength values
    """
    plt.figure(figsize=(10, 6))
    for i in range(endmembers.shape[0]):
        plt.plot(wavelengths, endmembers[i], label=f"{algorithm_name} - EM {i+1}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_amaps(abundance_map, H_t, wavelengths,ea, target_wl=750): 
    """
    abundance_map: numpy array of shape (rows, cols, n_em)
    H_t: hyperspectral cube of shape (rows, cols, bands)
    wavelengths: 1D array of length bands
    target_wl: wavelength (nm) to use for the background image
    """
    # figure out how many EMs
    rows, cols, n_em = abundance_map.shape

    # pick background band nearest target_wl
    band_idx = np.argmin(np.abs(wavelengths - target_wl))
    background = H_t[:, :, band_idx]

    # make subplots
    fig, axes = plt.subplots(1, n_em, figsize=(4 * n_em, 4), squeeze=False)

    for i in range(n_em):
        ax = axes[0, i]

        # plot background
        ax.imshow(background, cmap='gray')

        # overlay the i-th abundance map
        amap = abundance_map[:, :, i]
        im = ax.imshow(
            amap,
            cmap='inferno',
            alpha=0.6,
            vmin=0, vmax=1
        )

        ax.set_title(f"EM {i+1} Abundance")
        ax.axis('off')

        # colorbar for this subplot
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Abundance')

    plt.suptitle(f"Abundance Maps over {wavelengths[band_idx]:.0f} nm Background for {ea}", y=1.02)
    plt.tight_layout()
    plt.show()
    

def extract_endmembers(H_t: np.ndarray, 
                       wavelengths: np.ndarray, 
                       algorithm: Literal['nfindr', 'vca', 'fippi', 'atgp'],
                       n_endmembers: int = 5, 
                       show_endmembers: bool = True,
                       show_amaps: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract endmembers from a hyperspectral cube using specified algorithm.

    Args:
        H_t: (rows, cols, bands) hyperspectral data cube
        wavelengths: (bands,) array of wavelength values
        algorithm: 'nfnd', 'vca', 'fippi', or 'atgp'
        n_endmembers: number of endmembers to extract
        show_results: plot the extracted endmembers
        show_amaps : plot abundance maps if True
    
    Returns:
        endmembers: (n_endmembers, bands) array of extracted endmember spectra
        abundance_maps: (rows, cols, n_endmembers) array of abundance maps if plot_amaps is True
    """
    if algorithm == 'nfindr':
        ee_nfindr = eea.NFINDR()
        em_nfindr = ee_nfindr.extract(M=H_t, q=n_endmembers)
        # print("NFINDR's endmembers: ", em_nfindr)
        if show_endmembers:
            plot_endmembers(em_nfindr, wavelengths, title=f"NFINDR Endmember Spectra", algorithm_name="NFINDR")

        if show_amaps:
            fcls = amp.FCLS()
            amap_nfindr = fcls.map(H_t, em_nfindr, normalize=True)
            plot_amaps(np.array(amap_nfindr), H_t, wavelengths,"NFINDR",target_wl=750)
        return em_nfindr, amap_nfindr if show_amaps else em_nfindr

    elif algorithm == 'vca':
        image = rp.SpectralImage(H_t, spectral_axis=wavelengths)
        vca = rp.analysis.unmix.VCA(n_endmembers=n_endmembers, abundance_method='fcls')
        abundance_maps_vca, em_vca = vca.apply(image)
        em_vca_np = np.array(em_vca) #vca's EMs
        
        if show_endmembers:
            plot_endmembers(em_vca_np, wavelengths, title="VCA Endmember Spectra", algorithm_name="VCA")

        if show_amaps:
            fcls = amp.FCLS()
            amap = fcls.map(H_t,em_vca_np)
            plot_amaps(np.array(amap), H_t, wavelengths, "VCA", target_wl=750) 
        return em_vca_np, amap if show_amaps else em_vca_np
    
    elif algorithm == 'fippi':
        ee_fippi = eea.FIPPI()
        em_fippi = ee_fippi.extract(M=H_t, q=4)
        if show_endmembers:
            plot_endmembers(em_fippi, wavelengths, title="FIPPI Endmember Spectra", algorithm_name="FIPPI")
        
        if show_amaps:
            fcls = amp.FCLS()
            amap_fippi = fcls.map(H_t, em_fippi, normalize=True)
            plot_amaps(np.array(amap_fippi), H_t, wavelengths, "FIPPI", target_wl=750)
        return em_fippi, amap_fippi if show_amaps else em_fippi

    elif algorithm == 'atgp':
        atgp = eea.ATGP()
        em_atgp = atgp.extract(M=H_t, q=n_endmembers)
        if show_endmembers:
            plot_endmembers(em_atgp, wavelengths, title="ATGP Endmember Spectra", algorithm_name="ATGP")
        
        if show_amaps:
            fcls = amp.FCLS()
            amap_atgp = fcls.map(H_t, em_atgp, normalize=True)
            plot_amaps(np.array(amap_atgp), H_t, wavelengths, "ATGP", target_wl=750)
        return em_atgp, amap_atgp if show_amaps else em_atgp
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Choose from 'nfindr', 'vca', 'fippi', or 'atgp'.")

#%%
if __name__ == "__main__":
    # Example usage
    refl_cube_path = '/teamspace/studios/this_studio/isro-spectral-unmixing/data/den_reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz'

    unloaded = np.load(refl_cube_path)
    H = unloaded['den_refl_data']
    wavelengths = unloaded['wavelengths']
    H_t = np.moveaxis(H, 0, 2)  # Shape: (rows, cols, bands)
    H_t = H_t.astype('float32')
    ems, amap = extract_endmembers(H_t, wavelengths, algorithm='nfindr', n_endmembers=4)
    plot_endmembers(ems, wavelengths, title="N-FINDR Endmember Spectra", algorithm_name="N-FINDR")
    
    #DONT normalize the cube, it distorts the results apparently
    # rows, cols, bands = H_t.shape
    # X_flat = H_t.reshape(rows*cols, bands)
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # X_flat_norm = scaler.fit_transform(X_flat)
    # H_t_norm = X_flat_norm.reshape(rows, cols, bands)  # Reshape back to (rows, cols, bands)
    
    # ems_norm, amap_norm = extract_endmembers(H_t_norm, wavelengths, algorithm='nfindr', n_endmembers=4)
    # plot_endmembers(ems_norm, wavelengths, title="N-FINDR Endmember Spectra (Normalized)", algorithm_name="N-FINDR (Normalized)")

# %%
