from osgeo import gdal
from pathlib import Path
import numpy as np
import os 
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
import pysptools
from pysptools import eea
import gc
from skimage.restoration import denoise_tv_chambolle
from skimage.metrics import peak_signal_noise_ratio as psnr
import ramanspy as rp
from ramanspy.analysis.unmix import VCA
from pysptools import abundance_maps as amp
from sklearn.decomposition import PCA

files_info = []
data_dir = Path("/kaggle/input/iirs-data-2") # from here: https://www.kaggle.com/datasets/milapp180/iirs-data-2?select=iirs+data+bundle

for file_path in data_dir.rglob("*.qub"):
    qub_path = str(file_path)
    qub_stem = file_path.stem

    #get xml file
    meta_file = str(next(file_path.parent.glob("*.xml")))

    #get oat file
    oat_file = str(next(data_dir.rglob("*.oat")))
    
    files_info.append([qub_path, qub_stem, meta_file, oat_file])

def open_and_geotiff(file_info):
    data_cube = gdal.Open(file_info[0])
    # gdal.Translate(file_info[1], data_cube, format="GeoTiFF")
    data_arr = data_cube.ReadAsArray()
    return data_arr

def filter_validbands(data_arr):
    xls = pd.ExcelFile("/kaggle/input/iirs-hsi-data/iir/iir/miscellaneous/ch2_iirs_band_validity_list.xls")
    bands_validity = pd.read_excel(xls, 'Sheet2')
    wavelengths = bands_validity[bands_validity["E1G2"]=="VALID"]["Band Center Wavelength (nm)"].to_numpy()

    valid_bands = bands_validity[bands_validity["E1G2"]=="VALID"]["Band Number"].to_numpy()
    
    valid_bands_indices = valid_bands - 1
    data_arr_valid = data_arr[valid_bands_indices, :, :]

    #remove infrared bands
    return wavelengths[:109], data_arr_valid[:109, :, :]
    
def read_oat_solar_zenith(oat_path):
    """
    Parses the .oat file from the miscellaneous folder and extract the Solar Zenith Angle (field 31).
    From the readme.txt, the field 31 starts from 522nd byte (1-based, so 521 for python, 0-based), as previous 30 fields make for the 521 bytes

    Parameters
    ----------
    oat_path : str
        Path to the ASCII .oat file.

    Returns
    -------
    numpy.ndarray
        1D array of solar zenith angles (degrees), one per record.
    """
    sza_values = []
    with open(oat_path, 'r') as f:
        for line in f:
            # skip empty or too-short lines
            if len(line) < 530:
                continue
            raw = line[521:530]
            try:
                angle = float(raw)
            except ValueError:
                # e.g. header or malformed; append NaN
                angle = np.nan
            sza_values.append(angle)

    return np.array(sza_values)
    
def read_oat_timestamps(oat_path):

    timestamps = []

    with open(oat_path, 'r') as f:
        for line in f:
            if len(line) < 47:
                continue
            raw_time = line[18:47]
            parts = [raw_time[i:i+4].strip() for i in range(0, 28, 4)]
            # print(raw_time)
            # print(parts)
            year, month, day, hour, minute, sec, milisec = map(int, parts)

            #build datetime
            dt = datetime(year, month, day,
                         hour, minute, sec,
                         milisec*1000) #conv milisec to microsec

            timestamps.append(dt.timestamp())

    return timestamps

def save_data(file_info, data_cube):
    file_name = "reflectance__" + str(file_info[1]) + ".npz"
    np.savez_compressed(file_name, refl_data=data_cube)

def rad_to_refl(rad_data, file_info, wavelengths):

    xml_path = file_info[2]
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {'pds': 'http://pds.nasa.gov/pds4/pds/v1',
          'isda': 'https://isda.issdc.gov.in/pds4/isda/v1'}

    # acquisition time (UTC)
    t_str = root.find('.//pds:start_date_time', ns).text  # e.g. "2019-12-08T08:14:15.9609Z"
    t_end = root.find('.//pds:stop_date_time', ns).text
    t0 = datetime.fromisoformat(t_str.rstrip('Z'))
    t1 = datetime.fromisoformat(t_end.rstrip('Z'))

    n_lines = rad_data.shape[1]

    sza_array = read_oat_solar_zenith(file_info[3])
    oat_timestamps = read_oat_timestamps(file_info[3])

    image_times = np.linspace(t0.timestamp(), t1.timestamp(), n_lines) #times at which the images were captured
    sza_values_interp = np.interp(image_times, oat_timestamps, sza_array) #linear interpolation to find values of sza at the image captured times

    cos_sza = np.cos(np.radians(sza_values_interp))
    cos_sza = np.reshape(cos_sza, (1,-1,1))

    solar_flux_data = pd.read_csv("/kaggle/input/iirs-hsi-data/iir/iir/miscellaneous/ch2_iirs_solar_flux.txt", sep='\s+', header=None)
    solar_flux_data[0] = solar_flux_data[0].round(1)  # Round wavelength column

    solar_irradiance = solar_flux_data[solar_flux_data[0].isin(wavelengths)][1].to_numpy()
    solar_irradiance = np.reshape(solar_irradiance, (-1, 1,1))
    # print(solar_irradiance.shape)
    # print(cos_sza.shape)
    refl_data = ((np.pi * rad_data)/(cos_sza*solar_irradiance))
    
    return refl_data

def pipeline(file_info):

    results = {}
    
    rad_data_cube = open_and_geotiff(file_info)
    
    wavelengths, valid_rad_data = filter_validbands(rad_data_cube)
    
    del rad_data_cube
    gc.collect()
    print("valid bands extracted")
    
    reflectance_data = rad_to_refl(valid_rad_data, file_info, wavelengths)[:, :1000, :]
    print("rad to refl done")
    
    refl_data_denoise = denoise_tv_chambolle(reflectance_data ,max_num_iter=50, weight=20)
    print("DOne!")
    
    plot_denoised(reflectance_data, refl_data_denoise, wavelengths, file_info)
    dr = reflectance_data.max() - reflectance_data.min()
    print(f"PSNR for {file_info[1]}: {psnr(reflectance_data, refl_data_denoise, data_range=dr)}")
    
    bands, rows, cols = refl_data_denoise.shape
    H_t = refl_data_denoise.transpose(1, 2, 0)
    results["H_t"] = H_t
    
    del refl_data_denoise
    gc.collect()