import numpy as np
import torch
import math
from time import sleep
from dimension_reduction.ss_vae.config import get_config
from dimension_reduction.ss_vae.metrics_logger import MetricsLogger
from dimension_reduction.ss_vae.visualization import plot_losses
from .vae import VAE
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.restoration import denoise_tv_chambolle
from tqdm import tqdm

def main():
    """
    Main function to train the Variational Autoencoder (VAE) on reflectance data.
    It loads the data, preprocesses it, initializes the model, and runs the training loop.
    """
    # Load and preprocess the reflectance data
    refl_cube_path = '/teamspace/studios/this_studio/isro-spectral-unmixing/data/den_reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz' #the denoised image
    unloaded = np.load(refl_cube_path)
    refl_data = unloaded['den_refl_data']
    wavelengths = unloaded['wavelengths']
    print("Denoised image extracted!")

    spectral_bands = refl_data.shape[0]
    H_t = np.moveaxis(refl_data, 0, 2)

    rows, cols, bands = H_t.shape
    X_flat = H_t.reshape(rows*cols, bands)

    X_flat_train, X_flat_test = train_test_split(X_flat, train_size=0.8, shuffle=True)
    scaler = StandardScaler()
    X_flat_train_norm = scaler.fit_transform(X_flat_train)
    X_flat_test_norm = scaler.transform(X_flat_test)

    config = get_config()
    metrics = MetricsLogger()

    train_data = TensorDataset(torch.tensor(X_flat_train_norm))
    test_data = TensorDataset(torch.tensor(X_flat_test_norm))
    train_dl = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    test_dl = DataLoader(test_data, batch_size=config.batch_size, shuffle=True, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE(
        input_dim=spectral_bands,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim
    ).to(device)

    model = torch.compile(model)  # For PyTorch 2.0+ graph optimizations
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=config.scheduler_patience, threshold=0.01)
    patience_cntr = 0
    best_model_state = None
    best_test_loss = float('inf')
    recon_loss = torch.nn.HuberLoss()
    print("The Training Begins !")
    sleep(0.5)

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
            recon = model.decode(z)

            recon_loss_term = recon_loss(recon, x)
            kl_pd = 0.5 * (torch.exp(log_var) + mean**2 - 1 - log_var)  # shape (B, L)
            # clamp each dimension to at least free_bits
            kl_fb = torch.clamp(kl_pd, min=config.free_bits)
            kl_loss = kl_fb.sum(dim=1).mean()    
            loss = recon_loss_term + config.beta * kl_loss

            loss.backward()
            optim.step()

            metrics.update('train', loss.item(), recon_loss_term.item(), kl_loss.item())
            if i % config.update_interval == 0:
                train_pbar.set_postfix({
                    "loss": "{:.4f}".format(loss.item()),
                    "reconstruction": "{:.4f}".format(recon_loss_term.item()),
                    "kl": "{:.4f}".format(kl_loss),
                })
                if math.isnan(loss.item()):
                    raise ValueError("Loss went to nan.")

        # EVALUATION
        model.eval()
        test_pbar = tqdm(test_dl, total=len(test_dl), desc=f"Epoch {epoch+1}/{config.epochs} [Test]")
        with torch.no_grad():
            for x in test_pbar:
                x = x[0].float().to(device)

                mean, log_var = model.encode(x)
                z = model.sample(mean, log_var)
                recon = model.decode(z)

                recon_loss_term = recon_loss(recon, x)
                kl_pd = 0.5 * (torch.exp(log_var) + mean**2 - 1 - log_var)  # shape (B, L)
            # clamp each dimension to at least free_bits
                kl_fb = torch.clamp(kl_pd, min=config.free_bits)
                kl_loss = kl_fb.sum(dim=1).mean()    
                loss = recon_loss_term + config.beta * kl_loss

                metrics.update('val', loss.item(), recon_loss_term.item(), kl_loss.item())
                test_pbar.set_postfix({
                    "loss": "{:.4f}".format(loss.item()),
                    "reconstruction": "{:.4f}".format(recon_loss_term.item()),
                    "kl": "{:.4f}".format(kl_loss.item()),
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
    plot_losses(metrics, "vae_loss_plots")
    return model, best_model_state,  metrics.history    

if __name__ == "__main__":
    model, best_model_state, history = main()
   
    print("Training complete. Best model state saved.")
    # Save the best model state if needed
    import os
    from datetime import datetime
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    torch.save(model, f'models/vae_model_{timestamp}.pth')
    print("Best model saved to 'models'.")