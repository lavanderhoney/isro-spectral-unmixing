from time import sleep
from .config import get_config # import config from the same directory
from .metrics_logger import MetricsLogger
from .spatial_spectral_vae import SpatialSpectralNet
from .dataloaders import get_dataloaders, spectral_bands, neighborhood_size
from .visualization import plot_losses
from .utils import extract_spectral_data
from tqdm import tqdm
import math
import torch
import torch.optim as optim
import torch.nn as nn
#%%
#--------- TO-DO -----------
# change the dataloaders  accepts the data path, batch size, patch size from the user
#---------------------------

def main():
    config = get_config()   
    metrics = MetricsLogger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, test_dl = get_dataloaders(config.batch_size) 

    print("Dataloaders created !")
    model = SpatialSpectralNet(
        train_dl.dataset.__getattribute__('B'),  # number of spectral bands
        config.patch_size,  
        config.latent_dim,
        config.hidden_dim
    ).to(device)

    print("The Training Begins !")
    sleep(0.5)

    early_stop_patience=config.early_stop

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.scheduler_patience, threshold=0.01)
    patience_cntr = 0
    best_model_state = None
    # best_test_loss = float('inf')
    best_test_loss = float('inf')
    recon_loss = nn.MSELoss()

    for epoch in range(config.epochs):

        metrics.reset_epoch()
        # TRAINING
        model.train()
        train_pbar = tqdm(train_dl, total=len(train_dl), desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        for i, x in enumerate(train_pbar):
            x = x.float().to(device)

            optimizer.zero_grad()

            model.train()
            recon = model(x)
            input_spectra = extract_spectral_data(x)
            recon_loss_term = recon_loss(recon, input_spectra)

            kl_loss = model.encoder.kl_loss_term
            homology_loss = model.encoder.homology_loss_term

            loss = recon_loss_term + config.beta*kl_loss + homology_loss #beta-VAE

            loss.backward()
            optimizer.step()

            # update metrics
            metrics.update('train', loss.item(), recon_loss_term.item(), kl_loss.item(), homology_loss.item())

            if i % config.update_interval == 0:
                train_pbar.set_postfix({
                    "loss": "{:.4f}".format(loss.item()),
                    "reconstruction": "{:.4f}".format(recon_loss_term.item()),
                    "kl": "{:.4f}".format(kl_loss),
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
                recon = model(x)
                input_spectra = extract_spectral_data(x)
                recon_loss_term = recon_loss(recon, input_spectra)

                kl_loss = model.encoder.kl_loss_term
                homology_loss = model.encoder.homology_loss_term

                loss = recon_loss_term + config.beta*kl_loss + homology_loss #beta-VAE
                metrics.update('val', loss.item(), recon_loss_term.item(), kl_loss.item(), homology_loss.item())

            test_pbar.set_postfix({
                    "loss": "{:.4f}".format(loss.item()),
                    "reconstruction": "{:.4f}".format(recon_loss_term.item()),
                    "kl": "{:.4f}".format(kl_loss),
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
    plot_losses(metrics, 'loss_plots.png')
    return model, best_model_state, metrics

if __name__ == "__main__":
    model, best_model_statem, metrics = main()
    
    #save the model
    torch.save(model, "models/model_ss_vae.pth")