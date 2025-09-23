import os
from typing import Tuple, Optional

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

#%%
try:
	from src.dimension_reduction.ss_vae.dataloaders import open_datacube
except Exception:
	# Fallback import path if used as a package module
	from dimension_reduction.ss_vae.dataloaders import open_datacube


def spectral_fft_features(H: np.ndarray, keep: int = 16) -> np.ndarray:
	"""
	Compute FFT magnitude features along the spectral axis for each pixel.

	Input H must be (H, W, B). Returns features shaped (H*W, keep).
	keep controls how many lowest-frequency magnitudes to keep (excluding DC if desired).
	"""
	if H.ndim != 3:
		raise ValueError(f"Expected H with shape (H, W, B), got {H.shape}")
	h, w, b = H.shape

	# Flatten spatial dims -> (N, B)
	X = H.reshape(-1, b)

	# Remove per-pixel mean to reduce DC dominance
	X_center = X - X.mean(axis=1, keepdims=True)

	# FFT along spectral bands
	F = np.fft.rfft(X_center, axis=1)
	mag = np.abs(F)

	# Keep lowest frequencies (including DC after centering ~0)
	k = min(keep, mag.shape[1])
	feats = mag[:, :k]
	return feats


def extract_endmembers_fft(H: np.ndarray, n_endmembers: int = 4, keep_freqs: int = 16,
						   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Find endmembers by clustering FFT spectral features.

	- H: hyperspectral cube (H, W, B)
	- n_endmembers: number of clusters/endmembers
	- keep_freqs: number of low-frequency FFT magnitudes to use

	Returns (endmembers, label_map):
	  - endmembers: array (n_endmembers, B) averaged spectra per cluster
	  - label_map: array (H, W) of cluster indices
	"""
	if H.ndim != 3:
		raise ValueError("H must be (H, W, B)")
	h, w, b = H.shape

	feats = spectral_fft_features(H, keep=keep_freqs)

	# KMeans over features
	km = KMeans(n_clusters=n_endmembers, n_init=10, random_state=random_state)
	labels = km.fit_predict(feats)
	label_map = labels.reshape(h, w)

	# Compute average spectrum per cluster as the endmember
	X = H.reshape(-1, b)
	endmembers = np.zeros((n_endmembers, b), dtype=X.dtype)
	for k in range(n_endmembers):
		mask = labels == k
		if np.any(mask):
			endmembers[k] = X[mask].mean(axis=0)
		else:
			# If a cluster is empty (rare), fall back to cluster center mapped to spectral domain
			# Use nearest pixel to cluster center in feature space
			center = km.cluster_centers_[k]
			idx = np.argmin(np.linalg.norm(feats - center[None, :], axis=1))
			endmembers[k] = X[idx]

	return endmembers, label_map


def _save_outputs(base_dir: str, endmembers: np.ndarray, label_map: np.ndarray, prefix: str = "fft") -> None:
	os.makedirs(base_dir, exist_ok=True)
	np.save(os.path.join(base_dir, f"{prefix}_endmembers.npy"), endmembers)
	np.save(os.path.join(base_dir, f"{prefix}_labels.npy"), label_map)

def plot_endmembers(endmembers: np.ndarray, wavelengths: Optional[np.ndarray] = None,
					title: str = "Endmember Spectra", figsize: Tuple[int, int] = (10, 6)) -> None:
	"""
	Plot the endmember spectra in the spectral (wavelength) domain.
	endmembers: (K, B)
	wavelengths: (B,) optional x-axis; if None, uses band indices.
	"""
	if endmembers.ndim != 2:
		raise ValueError("endmembers must be (K, B)")
	k, b = endmembers.shape
	x = np.arange(b) if wavelengths is None else wavelengths
	plt.figure(figsize=figsize)
	for i in range(k):
		plt.plot(x, endmembers[i], label=f"Endmember {i+1}")
	plt.xlabel("Wavelength" if wavelengths is not None else "Band Index")
	plt.ylabel("Reflectance")
	plt.title(title)
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.show()


def plot_endmembers_fft(endmembers: np.ndarray, remove_mean: bool = True,
						title: str = "Endmember FFT magnitude", figsize: Tuple[int, int] = (10, 6)) -> None:
	"""
	Plot rFFT magnitude of each endmember spectrum.
	"""
	if endmembers.ndim != 2:
		raise ValueError("endmembers must be (K, B)")
	k, b = endmembers.shape
	X = endmembers.copy()
	if remove_mean:
		X = X - X.mean(axis=1, keepdims=True)
	F = np.fft.rfft(X, axis=1)
	mag = np.abs(F)
	freqs = np.arange(mag.shape[1])
	plt.figure(figsize=figsize)
	for i in range(k):
		plt.plot(freqs, mag[i], label=f"Endmember {i+1}")
	plt.xlabel("Frequency index (spectral)")
	plt.ylabel("Magnitude")
	plt.title(title)
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.show()


def plot_fft_feature_means(H: np.ndarray, label_map: np.ndarray, keep_freqs: int = 16,
						   title: str = "Cluster-mean FFT features",
						   figsize: Tuple[int, int] = (10, 6)) -> None:
	"""
	Plot the mean FFT feature vector per cluster.
	H: (H, W, B)
	label_map: (H, W) cluster indices from extract_endmembers_fft
	"""
	if H.ndim != 3 or label_map.ndim != 2:
		raise ValueError("H must be (H,W,B) and label_map (H,W)")
	h, w, _ = H.shape
	feats = spectral_fft_features(H, keep=keep_freqs)
	labels = label_map.reshape(-1)
	k = labels.max() + 1
	means = []
	for i in range(k):
		m = feats[labels == i].mean(axis=0)
		means.append(m)
	means = np.stack(means, axis=0)
	x = np.arange(keep_freqs)
	plt.figure(figsize=figsize)
	for i in range(k):
		plt.plot(x, means[i], label=f"Cluster {i}")
	plt.xlabel("Frequency index (kept)")
	plt.ylabel("FFT magnitude")
	plt.title(title)
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.show()


def plot_cluster_map(label_map: np.ndarray,
					 title: str = "Endmember Cluster Map",
					 figsize: Tuple[int, int] = (8, 6),
					 save_path: Optional[str] = None) -> None:
	"""
	Plot the cluster label map with a discrete colormap.
	label_map: (H, W) integer labels starting at 0.
	"""
	if label_map.ndim != 2:
		raise ValueError("label_map must be (H, W)")
	k = int(label_map.max()) + 1
	base = plt.cm.get_cmap('tab20') if k <= 20 else plt.cm.get_cmap('gist_ncar')
	colors = base(np.linspace(0, 1, k))
	cmap = ListedColormap(colors[:k])
	norm = BoundaryNorm(np.arange(-0.5, k + 0.5, 1), cmap.N)

	plt.figure(figsize=figsize)
	im = plt.imshow(label_map, cmap=cmap, norm=norm, interpolation='nearest')
	cbar = plt.colorbar(im, ticks=np.arange(0, k, 1))
	cbar.set_label('Cluster')
	plt.title(title)
	plt.axis('off')
	plt.tight_layout()
	if save_path:
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		plt.savefig(save_path, dpi=200)
	plt.show()

#%%
if __name__ == "__main__":
    # Example usage
    data_path = "/teamspace/studios/this_studio/isro-spectral-unmixing/data/den_reflectance_ch2_iir_nci_20191208T1407123802_d_img_d18.npz"
    H, wavelengths = open_datacube(data_path)
    H_t = H.transpose(1, 2, 0)  # (H, W, B)

    endmembers, label_map = extract_endmembers_fft(H_t, n_endmembers=4, keep_freqs=16)

    # _save_outputs("outputs", endmembers, label_map, prefix="fft")

    plot_endmembers(endmembers, wavelengths=wavelengths, title="Endmember Spectra from FFT Clustering")
    plot_endmembers_fft(endmembers, title="Endmember FFT Magnitudes")
    plot_fft_feature_means(H_t, label_map, keep_freqs=16, title="Cluster-mean FFT Features")
#%%