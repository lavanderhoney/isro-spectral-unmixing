import numpy as np
import pandas as pd
from pathlib import Path

def match_relab_iirs_bands(relab_csv_path:str):
    xls = pd.ExcelFile("/teamspace/studios/this_studio/isro-spectral-unmixing/data/ch2_iirs_band_validity_list.xls")
    bands_validity = pd.read_excel(xls, 'Sheet2')
    valid_rows = bands_validity[bands_validity["E1G2"] == "VALID"]

    iirs_wavelengths = valid_rows["Band Center Wavelength (nm)"].to_numpy()[:109]  # ignore IR bands
    valid_band_numbers = valid_rows["Band Number"].to_numpy()[:109]

    # print("IIRS Valid Wavelengths (nm):", iirs_wavelengths.shape)

    # --- Load RELAB spectrum ---
    relab_data = pd.read_csv(relab_csv_path)
    relab_wavelengths = relab_data["Wavelength (nm)"].to_numpy()
    relab_reflectance = relab_data.iloc[:, 3].to_numpy()  # 4th column = reflectance

    # --- Find nearest RELAB wavelength for each IIRS band ---
    diff = np.abs(relab_wavelengths[:, None] - iirs_wavelengths)
    nearest_indices = diff.argmin(axis=0)
    nearest_distances = diff.min(axis=0)

    # --- Optional: filter only those within a reasonable tolerance (e.g., ±10 nm) ---
    tolerance_nm = 10
    valid_mask = nearest_distances <= tolerance_nm

    matched_relab_reflectance = relab_reflectance[nearest_indices[valid_mask]]
    matched_relab_wavelengths = relab_wavelengths[nearest_indices[valid_mask]]
    matched_iirs_wavelengths = iirs_wavelengths[valid_mask]
    matched_iirs_band_numbers = valid_band_numbers[valid_mask]

    # print(f"Matched bands: {matched_relab_reflectance.shape[0]} (within ±{tolerance_nm} nm)")

    return (
        matched_iirs_band_numbers,   # which IIRS bands you can use
        matched_iirs_wavelengths,    # filtered IIRS wavelengths
        matched_relab_wavelengths,   # matched RELAB wavelengths
        matched_relab_reflectance    # RELAB reflectance for training
    )

# match_relab_iirs_bands("/teamspace/studios/this_studio/isro-spectral-unmixing/data/RELAB data/15058/BDR/LRCMP172_15058_BrownPyroxene_Coarse_BDR.csv")
relab_dir = Path("/teamspace/studios/this_studio/isro-spectral-unmixing/data/RELAB data")
pyroxenes = list(relab_dir.rglob("L*Pyroxene*.csv")) # class 0
plagioclases = list(relab_dir.rglob("L*Plagioclase*.csv")) # class 1
olivines = list(relab_dir.rglob("L*Olivine*.csv")) # class 2
ilemnites = list(relab_dir.rglob("L*Ilmenite*.csv")) # class 3
print(f"Found {len(pyroxenes)} pyroxene samples")
print(f"Found {len(plagioclases)} plagioclase samples")
print(f"Found {len(olivines)} olivine samples")
print(f"Found {len(ilemnites)} ilmenite samples")

# get reflectance for all samples
all_samples = pyroxenes + plagioclases + olivines + ilemnites
label_mapping = {
    "pyroxene": 0,
    "plagioclase": 1,
    "olivine": 2,
    "ilmenite": 3
}
all_labels = []
all_reflectance = []
for sample in all_samples:
    _, _, _, reflectance = match_relab_iirs_bands(sample)
    if reflectance.shape[0] == 0:
        continue
    all_reflectance.append(reflectance)
    # Get the label from the mapping
    if "pyroxene" in sample.name.lower():
        all_labels.append(label_mapping["pyroxene"])
    elif "plagioclase" in sample.name.lower():
        all_labels.append(label_mapping["plagioclase"])
    elif "olivine" in sample.name.lower():
        all_labels.append(label_mapping["olivine"])
    elif "ilmenite" in sample.name.lower():
        all_labels.append(label_mapping["ilmenite"])
max_len = max(len(s) for s in all_reflectance)

padded = np.zeros((len(all_reflectance), max_len), dtype=np.float32)   # padded with 0
mask = np.zeros((len(all_reflectance), max_len), dtype=np.float32)     # 1.0 valid, 0.0 padded
all_labels = np.array(all_labels, dtype=np.int8)
for i, arr in enumerate(all_reflectance):
    L = len(arr)
    padded[i, :L] = arr
    mask[i, :L] = 1.0
    
print("All reflectance shape:", padded.shape)
print("All labels shape:", all_labels.shape)

#wrapper function
def get_relab_iirs_data():
    return padded, all_labels, mask