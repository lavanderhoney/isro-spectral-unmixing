"""
Integration script to extract endmembers using FFT method and classify them using the trained CNN.
"""
#%%
import numpy as np
from mineral_analysis.unmixing.fft import extract_endmembers_fft, plot_endmembers, plot_cluster_map
from mineral_analysis.endmember_extraction import extract_endmembers
from mineral_analysis.classification.spectral_cnn import load_model_and_predict
from dimension_reduction.ss_vae.dataloaders import open_datacube

def run_fft_classification_pipeline(em_extract_method:str, data_path: str, model_path: str = None, train_model: bool = True, n_endmembers: int = 4):
    """
    Complete pipeline: FFT endmember extraction + CNN classification.
    
    Args:
        data_path: Path to hyperspectral data cube
        model_path: Path to save/load CNN model
        train_model: Whether to train a new model or load existing
        n_endmembers: Number of endmembers to extract
    """
    if model_path is None:
        model_path = "/teamspace/studios/this_studio/isro-spectral-unmixing/models/spectral_cnn.pth"
    
    # print("=" * 60)
    # print("FFT Endmember Extraction + CNN Classification Pipeline")
    # print("=" * 60)
    
    # 1. Load hyperspectral data
    print("\nLoading hyperspectral data...")
    refl_data, wavelengths = open_datacube(data_path)
    H = np.moveaxis(refl_data, 0, 2)  # (H, W, B)
    print(f"Data shape: {H.shape}")
    
    # 2. Extract endmembers using FFT
    if em_extract_method.lower() == "fft":
        print("\nExtracting endmembers using FFT method...")
        endmembers, label_map = extract_endmembers_fft(H, n_endmembers=n_endmembers, keep_freqs=24)
    elif em_extract_method.lower() == "vca":
        print("\nExtracting endmembers using VCA method...")
        unloaded = np.load(data_path)
        wavelengths = unloaded['wavelengths']
        endmembers, label_map = extract_endmembers(H, algorithm="vca", wavelengths=wavelengths, n_endmembers=n_endmembers, show_amaps=True)
    print(f"Extracted {endmembers.shape[0]} endmembers with {endmembers.shape[1]} spectral bands")
    
    #  load CNN model
    print(f"\nLoading pre-trained CNN from {model_path}...")
    
    # 4. Classify endmembers
    print("\nClassifying endmembers...")
    pred_classes, probabilities, mineral_mapping = load_model_and_predict(model_path, endmembers)
    
    # 5. Visualize results
    print("\n5. Generating visualizations...")
    
    # Plot endmember spectra
    if wavelengths is not None:
        plot_endmembers(endmembers, wavelengths=wavelengths, title="FFT-Extracted Endmember Spectra")
    else:
        plot_endmembers(endmembers, title="FFT-Extracted Endmember Spectra")
    
    # Plot cluster map
    # plot_cluster_map(label_map, title="FFT Endmember Cluster Map")
    
    # # 6. Summary
    # print("\n6. Classification Summary:")
    # print("=" * 40)
    # for i, pred_class in enumerate(pred_classes):
    #     mineral_name = mineral_mapping[pred_class]
    #     confidence = probabilities[i, pred_class] * 100
    #     print(f"Endmember {i+1}: {mineral_name.upper()} ({confidence:.1f}% confidence)")
    print("=" * 60)
    return endmembers, label_map, pred_classes, probabilities, mineral_mapping


if __name__ == "__main__":
    # Example usage
    all_data_paths = ["/teamspace/studios/this_studio/isro-spectral-unmixing/data/den_reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz",
                      "/teamspace/studios/this_studio/isro-spectral-unmixing/data/den_reflectance_ch2_iir_nci_20191208T1407123802_d_img_d18.npz",
                      "/teamspace/studios/this_studio/isro-spectral-unmixing/data/den_reflectance_ch2_iir_nci_20210620T2110364457_d_img_hw1 (1).npz",
                      "/teamspace/studios/this_studio/isro-spectral-unmixing/data/den_reflectance_ch2_iir_nci_20211214T2146149094_d_img_hw1.npz"
                      ]
    # for i, path in enumerate(all_data_paths):
    #     print("\n" + "="*80)
    #     print(f"{i+1}. {path}")
    #     results = run_fft_classification_pipeline(
    #         em_extract_method="fft",
    #         data_path=path,
    #         model_path="/teamspace/studios/this_studio/isro-spectral-unmixing/models/SpectralCNN.pth",
    #         train_model=False,  # Set to False if you have a pre-trained model
    #         n_endmembers=4
    #     )
    
    #     endmembers, label_map, pred_classes, probabilities, mineral_mapping = results

    # print("\nPipeline completed successfully with FFT!")
        
    print("\n\n" + "="*80 + "\n\n")
    print("Now running VCA + CNN pipeline...\n")
    for i, path in enumerate(all_data_paths):
        print("\n" + "="*80)
        print(f"{i+1}. {path}")
        results = run_fft_classification_pipeline(
            em_extract_method="vca",
            data_path=path,
            model_path="/teamspace/studios/this_studio/isro-spectral-unmixing/models/SpectralCNN.pth",
            train_model=False,  # Set to False if you have a pre-trained model
            n_endmembers=4
        )
    
        endmembers, label_map, pred_classes, probabilities, mineral_mapping = results

        print("\nPipeline completed successfully with VCA!")
#%%