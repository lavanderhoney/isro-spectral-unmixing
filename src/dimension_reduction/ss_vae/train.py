from time import sleep
from typing import Tuple
from dimension_reduction.ss_vae.config import get_config # import config from the same directory
from dimension_reduction.ss_vae.metrics_logger import MetricsLogger
from dimension_reduction.ss_vae.spatial_spectral_vae import SpatialSpectralNet
from dimension_reduction.ss_vae.dataloaders import get_dataloaders_ssvae
from dimension_reduction.ss_vae.visualization import plot_losses
from dimension_reduction.ss_vae.utils import extract_spectral_data
from dimension_reduction.ss_vae.dataloaders import get_dataloaders, open_datacube
from mineral_analysis.endmember_extraction import extract_endmembers
from mineral_analysis.unmixing.ae import spectral_angle_distance_loss, spectral_information_divergence_loss, entropy_loss
from tqdm import tqdm
import math
import torch
import torch.optim as optim
import torch.nn as nn
#%%
#--------- TO-DO -----------
#---------------------------

def main(config):
    metrics = MetricsLogger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    refl_cube_path = '/teamspace/studios/this_studio/isro-spectral-unmixing/data/den_reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz' #the denoised image
    train_dl, test_dl, wavelengths = get_dataloaders(refl_cube_path)
    H, wavelengths = open_datacube(refl_cube_path)
    H_t = H.transpose(1, 2, 0)
    train_dl, test_dl = get_dataloaders_ssvae(config.data_path, config.batch_size, config.patch_size, config.test_size) 
    print("Dataloaders created !")
    #%%
    ems, _ = extract_endmembers(H_t, wavelengths, algorithm='vca', n_endmembers=4, show_endmembers=False, show_amaps=False)
    print("VCA done")
    model = SpatialSpectralNet(
        train_dl.dataset.__getattribute__('B'),  # number of spectral bands
        config.patch_size,  
        4,
        config.hidden_dim,
        em_spectra=ems,
        lstm_layers=config.lstm_layers,
        cnn_layers=config.cnn_layers,
        free_bits=config.free_bits
    ).to(device)
    # Wrap in torch.compile for PyTorch 2.0+ graph optimizations
    model = torch.compile(model)  
    print("The Training Begins !")
    sleep(0.5)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.scheduler_patience, threshold=0.01)
    patience_cntr = 0
    best_model_state = None
    # best_test_loss = float('inf')
    best_test_loss = float('inf')
    # recon_loss = nn.MSELoss()
    w_recon, w_hm, w_sid, w_ent = 1, 1, 1, 0
    for epoch in range(config.epochs):

        metrics.reset_epoch()
        # TRAINING
        model.train()
        train_pbar = tqdm(train_dl, total=len(train_dl), desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        for i, x in enumerate(train_pbar):
            x = x.float().to(device)

            optimizer.zero_grad()

            model.train()
            z, mean, recon = model(x)
            input_spectra = extract_spectral_data(x)
            
            recon_loss_term = spectral_information_divergence_loss(recon, input_spectra)
            sid_loss_term = spectral_angle_distance_loss(recon, input_spectra)
            entropy_loss_term = entropy_loss(z)
            kl_loss = model.encoder.kl_loss_term
            homology_loss = model.encoder.homology_loss_term

            loss = w_recon*recon_loss_term + w_sid*sid_loss_term - w_ent*entropy_loss_term + config.beta * kl_loss + w_hm*homology_loss
            loss.backward()
            optimizer.step()

            # update metrics
            metrics.update('train', loss.item(), recon_loss_term.item(), kl_loss.item(), homology_loss.item())

            if i % config.update_interval == 0:
                train_pbar.set_postfix({
                    "loss": "{:.4f}".format(loss.item()),
                    "reconstruction": "{:.4f}".format(recon_loss_term.item()),
                    "kl": "{:.4f}".format(kl_loss),
                    "sid_loss": "{:.4f}".format(sid_loss_term.item()),
                    "homology": "{:.4f}".format(homology_loss),
                })
                if math.isnan(loss.item()):
                    raise ValueError("Loss went to nan.")

        # EVAL
        model.eval()
        test_pbar = tqdm(test_dl, total=len(test_dl), desc=f"Epoch {epoch+1}/{config.epochs} [Eval ]")
        for x in test_pbar:
            x=x.float().to(device)
            with torch.inference_mode():
                z, mean, recon = model(x)
                input_spectra = extract_spectral_data(x)
                recon_loss_term = spectral_information_divergence_loss(recon, input_spectra)
                sid_loss_term = spectral_angle_distance_loss(recon, input_spectra)
                entropy_loss_term = entropy_loss(z)
                kl_loss = model.encoder.kl_loss_term
                homology_loss = model.encoder.homology_loss_term

                loss = w_recon*recon_loss_term + w_sid*sid_loss_term - w_ent*entropy_loss_term + config.beta * kl_loss + w_hm*homology_loss
                metrics.update('val', loss.item(), recon_loss_term.item(), kl_loss.item(), homology_loss.item())

            test_pbar.set_postfix({
                    "loss": "{:.4f}".format(loss.item()),
                    "reconstruction": "{:.4f}".format(recon_loss_term.item()),
                    "kl": "{:.4f}".format(kl_loss),
                    "sid_loss": "{:.4f}".format(sid_loss_term.item()),
                    "homology": "{:.4f}".format(homology_loss),
                })

        avg_train_loss = metrics.finalize_epoch('train')
        avg_test_loss = metrics.finalize_epoch('val')

        scheduler.step(avg_test_loss)
        if avg_test_loss < best_test_loss:
            print("New best model found")
            best_test_loss = avg_test_loss
            best_model_state = model.state_dict()
            patience_cntr = 0
        else:
            patience_cntr += 1
        if patience_cntr >= config.early_stop:
            print(f"Early stopping triggered at epoch: {epoch}")
            break
    plot_losses(metrics, 'ssvae_loss_plots')
    return model, metrics
#%%
if __name__ == "__main__":

    config = get_config()
    model, metrics = main(config)
    import os
    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    os.makedirs("models", exist_ok=True)
    
    #save the model with config
    state = {
        'model_state': model.state_dict(),
        'metrics': metrics,
        'config': config,
        'timestamp': timestamp
    }
    torch.save(state, f"models/model_state_ss_vae_{timestamp}.pth")