import numpy as np
import torch
from time import sleep
from dimension_reduction.ss_vae.config import get_config
from dimension_reduction.ss_vae.metrics_logger import MetricsLogger
from dimension_reduction.ss_vae.visualization import plot_losses
from dimension_reduction.ss_vae.dataloaders import get_dataloaders, open_datacube
from .vae import VAE
from mineral_analysis.unmixing.ae import spectral_angle_distance_loss, spectral_information_divergence_loss, entropy_loss
from mineral_analysis.endmember_extraction import extract_endmembers
from tqdm import tqdm

def main():
    """
    Main function to train the Variational Autoencoder (VAE) on reflectance data.
    It loads the data, preprocesses it, initializes the model, and runs the training loop.
    """
    # Load and preprocess the reflectance data
    refl_cube_path = '/teamspace/studios/this_studio/isro-spectral-unmixing/data/den_reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz' #the denoised image
    train_dl, test_dl, wavelengths = get_dataloaders(refl_cube_path)
    H, wavelengths = open_datacube(refl_cube_path)
    H_t = H.transpose(1, 2, 0)  
    # Get a batch to determine the number of spectral bands
    first_batch = next(iter(train_dl))
    n_bands = first_batch[0].shape[1]  # Number of spectral bands
    config = get_config()
    metrics = MetricsLogger()

    torch.autograd.set_detect_anomaly(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     # Initialize the decoder matrix E as endmembers from VCA
    ems, _ = extract_endmembers(H_t, wavelengths, algorithm='vca', n_endmembers=4, show_endmembers=False, show_amaps=False)
    print("VCA done")
    print(H_t.shape, wavelengths.shape, ems.shape) # type: ignore
    model = VAE(
        input_dim=n_bands,  # Number of bands
        hidden_dim=config.hidden_dim,
        latent_dim=4,
        em_spectra=ems,
    ).to(device)

    model = torch.compile(model)  # For PyTorch 2.0+ graph optimizations
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=config.scheduler_patience, threshold=0.01)
    patience_cntr = 0
    best_model_state = None
    best_test_loss = float('inf')
    print("The Training Begins !")
    sleep(0.5)

    w_recon, w_mv, w_sid, w_ent = 1, 1, 1, 0 # Weights for the losses
    for epoch in range(config.epochs):
        metrics.reset_epoch()

        # TRAINING
        model.train()
        train_pbar = tqdm(train_dl, total=len(train_dl), desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        for i, x in enumerate(train_pbar):
            x = x[0].float().to(device)  # Extract the tensor from the tuple

            optim.zero_grad()

            mean, log_var = model.encode(x)
            z = model.sample(mean, log_var)
            x_hat, E_positive = model.decode(z)

            recon_loss_term = spectral_angle_distance_loss(x_hat, x)
            sid_loss_term = spectral_information_divergence_loss(x_hat, x)
            entropy_loss_term = entropy_loss(z)
            kl_pd = 0.5 * (torch.exp(log_var) + mean**2 - 1 - log_var)  # shape (B, L)
            # clamp each dimension to at least free_bits
            kl_fb = torch.clamp(kl_pd, min=config.free_bits)
            kl_loss = kl_fb.sum(dim=1).mean()    
            loss = w_recon*recon_loss_term + w_sid*sid_loss_term - w_ent*entropy_loss_term + config.beta * kl_loss

            loss.backward()
            optim.step()

            metrics.update(phase='train', total=loss.item(), recon=recon_loss_term.item(), kl=kl_loss.item(), sid=sid_loss_term.item(), homology=entropy_loss_term.item()) # store entropy in metrics under homology as surrogate

            if i % config.update_interval == 0:
                train_pbar.set_postfix({
                    "loss": "{:.5f}".format(loss.item()),
                    "reconstruction": "{:.4f}".format(recon_loss_term.item()),
                    "kl_loss": "{:.4f}".format(kl_loss.item()),
                    "sid_loss": "{:.4f}".format(sid_loss_term.item()),
                    "entropy": "{:.4f}".format(entropy_loss_term.item())
                })

        # EVALUATION
        model.eval()
        test_pbar = tqdm(test_dl, total=len(test_dl), desc=f"Epoch {epoch+1}/{config.epochs} [Test]")
        with torch.no_grad():
            for x in test_pbar:
                x = x[0].float().to(device)

                mean, log_var = model.encode(x)
                z = model.sample(mean, log_var)
                x_hat, E_positive = model.decode(z)

                recon_loss_term = spectral_angle_distance_loss(x_hat, x)
                sid_loss_term = spectral_information_divergence_loss(x_hat, x)
                entropy_loss_term = entropy_loss(z)
                kl_pd = 0.5 * (torch.exp(log_var) + mean**2 - 1 - log_var)  # shape (B, L)
                # clamp each dimension to at least free_bits
                kl_fb = torch.clamp(kl_pd, min=config.free_bits)
                kl_loss = kl_fb.sum(dim=1).mean()    
                loss = w_recon*recon_loss_term + w_sid*sid_loss_term - w_ent*entropy_loss_term + config.beta * kl_loss

                metrics.update(phase='val', total=loss.item(), recon=recon_loss_term.item(), kl=kl_loss.item(), sid=sid_loss_term.item(), homology=entropy_loss_term.item()) # store entropy in metrics under homology as surrogate

                if i % config.update_interval == 0:
                    test_pbar.set_postfix({
                        "loss": "{:.5f}".format(loss.item()),
                        "reconstruction": "{:.4f}".format(recon_loss_term.item()),
                        "kl_loss": "{:.4f}".format(kl_loss.item()),
                        "sid_loss": "{:.4f}".format(sid_loss_term.item()),
                        "entropy": "{:.4f}".format(entropy_loss_term.item())
                    })


        avg_test_loss = metrics.finalize_epoch('val')
        scheduler.step(avg_test_loss)

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model_state = model.state_dict()
            patience_cntr = 0
        else:
            patience_cntr += 1

        if patience_cntr >= config.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    return model, best_model_state,  metrics   

if __name__ == "__main__":
    model, best_model_state, metrics = main()
   
    print("Training complete. Best model state saved.")
    # Save the best model state if needed
    import os
    from datetime import datetime
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    state = {
        'model_state': model.state_dict(),
        'metrics': metrics,
        'config': get_config(),
        'timestamp': timestamp
    }
    torch.save(state, f'models/model_state_vae_unmix_{timestamp}.pth')
    print("Best model saved to 'models'.")
    plot_losses(metrics, "vae_loss_plots")
