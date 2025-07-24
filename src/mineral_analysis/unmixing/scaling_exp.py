import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
refl_cube_path = '/teamspace/studios/this_studio/isro-spectral-unmixing/data/den_reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz'

unloaded = np.load(refl_cube_path)
H = unloaded['den_refl_data']
wavelengths = unloaded['wavelengths']

#%%
H_t = np.moveaxis(H, 0, 2)  # Shape: (rows, cols, bands)
H_t = H_t.astype('float32')
rows, cols, bands = H_t.shape
X_flat = H_t.reshape(rows*cols, bands)
print("Data loaded and reshaped:", X_flat.shape)

scaler = StandardScaler()
X_flat_norm = scaler.fit_transform(X_flat)

orig_s = (X_flat[0] - X_flat[0].min()) / (X_flat[0].max() - X_flat[0].min() + 1e-6)
recon_s = (X_flat_norm[0] - X_flat_norm[0].min()) / (X_flat_norm[0].max() - X_flat_norm[0].min() + 1e-6)
plt.plot(X_flat[0], '--', label='Original')
plt.plot(X_flat_norm[0], '-', label='Normalized', alpha=0.7)
plt.plot(recon_s, '-', label='Normalized min-maxed', alpha=0.7)
plt.plot(orig_s, '--', label='Original Min-maxed', alpha=0.5)
plt.title(f"Pixel {0} Spectra")
plt.xlabel("Band index")
plt.ylabel("Normalized intensity")
plt.legend()
plt.grid(True)
plt.show()

#%%
import seaborn as sns

# Sample a few (e.g., 100) random pixels to visualize their spectra
num_samples = 100
indices = np.random.choice(rows * cols, size=num_samples, replace=False)
sampled_spectra = X_flat_norm[indices]

plt.figure(figsize=(10, 6))
for i, spectrum in enumerate(sampled_spectra):
    plt.plot(spectrum, alpha=0.6, linewidth=0.7, color='blue')

plt.title("Sampled Pixel Spectral Signatures (Normalized)")
plt.xlabel("Band index")
plt.ylabel("Normalized intensity")
plt.grid(True)
plt.show()
#%%
import seaborn as sns

# Sample a few (e.g., 100) random pixels to visualize their spectra
num_samples = 100
indices = np.random.choice(rows * cols, size=num_samples, replace=False)
sampled_spectra = X_flat[indices]

plt.figure(figsize=(10, 6))
for i, spectrum in enumerate(sampled_spectra):
    plt.plot(spectrum, alpha=0.6, linewidth=0.7, color='blue')

plt.title("Sampled Pixel Spectral Signatures")
plt.xlabel("Band index")
plt.ylabel("Intensity")
plt.grid(True)
plt.show()

#%%
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
refl_cube_path = '/teamspace/studios/this_studio/isro-spectral-unmixing/data/den_reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz'

unloaded = np.load(refl_cube_path)
H = unloaded['den_refl_data']
wavelengths = unloaded['wavelengths']

#%%
H_t = np.moveaxis(H, 0, 2)  # Shape: (rows, cols, bands)
H_t = H_t.astype('float32')
rows, cols, bands = H_t.shape
X_flat = H_t.reshape(rows*cols, bands)
print("Data loaded and reshaped:", X_flat.shape)

scaler = StandardScaler()
X_flat_norm = scaler.fit_transform(X_flat)

orig_s = (X_flat[0] - X_flat[0].min()) / (X_flat[0].max() - X_flat[0].min() + 1e-6)
recon_s = (X_flat_norm[0] - X_flat_norm[0].min()) / (X_flat_norm[0].max() - X_flat_norm[0].min() + 1e-6)
plt.plot(X_flat[0], '--', label='Original')
plt.plot(X_flat_norm[0], '-', label='Normalized', alpha=0.7)
plt.plot(recon_s, '-', label='Normalized min-maxed', alpha=0.7)
plt.plot(orig_s, '--', label='Original Min-maxed', alpha=0.5)
plt.title(f"Pixel {0} Spectra")
plt.xlabel("Band index")
plt.ylabel("Normalized intensity")
plt.legend()
plt.grid(True)
plt.show()

#%%
import seaborn as sns

# Sample a few (e.g., 100) random pixels to visualize their spectra
num_samples = 100
indices = np.random.choice(rows * cols, size=num_samples, replace=False)
sampled_spectra = X_flat_norm[indices]

plt.figure(figsize=(10, 6))
for i, spectrum in enumerate(sampled_spectra):
    plt.plot(spectrum, alpha=0.6, linewidth=0.7, color='blue')

plt.title("Sampled Pixel Spectral Signatures (Normalized)")
plt.xlabel("Band index")
plt.ylabel("Normalized intensity")
plt.grid(True)
plt.show()
#%%
import seaborn as sns

# Sample a few (e.g., 100) random pixels to visualize their spectra
num_samples = 100
orig_s = (X_flat - X_flat.min()) / (X_flat.max() - X_flat.min() + 1e-6)
indices = np.random.choice(rows * cols, size=num_samples, replace=False)
sampled_spectra = orig_s[indices]

plt.figure(figsize=(10, 6))
for i, spectrum in enumerate(sampled_spectra):
    plt.plot(spectrum, alpha=0.6, linewidth=0.7, color='blue')

plt.title("Sampled Pixel Spectral Signatures (Original Min-maxed)")
plt.xlabel("Band index")
plt.ylabel("Intensity")
plt.grid(True)
plt.show()

#%%
sns.displot(X_flat[0], kde=True, bins=500)

#%%
sns.displot(orig_s[0], kde=True, bins=500)

#%%
sns.displot(X_flat_norm[0], kde=True, bins=50)