import numpy as np
import torch
import math
from time import sleep
from dimension_reduction.ss_vae.config import get_config
from dimension_reduction.ss_vae.metrics_logger import MetricsLogger
from dimension_reduction.ss_vae.visualization import plot_losses
from dimension_reduction.ss_vae.dataloaders import get_dataloaders
from mineral_analysis.unmixing.ae import AE, spectral_angle_distance_loss, total_variation
from tqdm import tqdm


def main():
    """
    Main function to train the LMM based Autoencoder (AE) on reflectance data.
    """
    # Load and preprocess the reflectance data
    refl_cube_path = '/teamspace/studios/this_studio/isro-spectral-unmixing/data/den_reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz' #the denoised image
    train_dl, test_dl, wavelengths = get_dataloaders(refl_cube_path)
    # Get a batch to determine the number of spectral bands
    first_batch = next(iter(train_dl))
    n_bands = first_batch[0].shape[1]  # Number of spectral bands
    config = get_config()
    metrics = MetricsLogger()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE(
        input_dim=n_bands,  # Number of bands
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim
    ).to(device)
    model = torch.compile(model)

    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=config.scheduler_patience, threshold=0.01)
    patience_cntr = 0
    best_model_state = None
    best_test_loss = float('inf')
    
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
           
            x_hat, z, E_positive = model(x)

            recon_loss_term = spectral_angle_distance_loss(x_hat, x)
            mv_loss_term = total_variation(E_positive)
            loss = recon_loss_term + config.gamma * mv_loss_term

            loss.backward()
            optim.step()

            metrics.update('train', loss.item(), recon_loss_term.item(), mv_loss_term.item())
            if i % config.update_interval == 0:
                train_pbar.set_postfix({
                    "loss": "{:.4f}".format(loss.item()),
                    "reconstruction": "{:.4f}".format(recon_loss_term.item()),
                    "total_variation": "{:.4f}".format(mv_loss_term.item() if mv_loss_term is not None else 0)
                })
                if math.isnan(loss.item()):
                    raise ValueError("Loss went to nan.")

        # EVALUATION
        model.eval()
        test_pbar = tqdm(test_dl, total=len(test_dl), desc=f"Epoch {epoch+1}/{config.epochs} [Test]")
        with torch.no_grad():
            for x in test_pbar:
                x = x[0].float().to(device)

                x_hat, z, E_positive = model(x)
                recon_loss_term = spectral_angle_distance_loss(x_hat, x)
                mv_loss_term = total_variation(E_positive)
                loss = recon_loss_term + config.gamma * mv_loss_term
            
                metrics.update('val', loss.item(), recon_loss_term.item(), mv_loss_term.item())
                test_pbar.set_postfix({
                    "loss": "{:.4f}".format(loss.item()),
                    "reconstruction": "{:.4f}".format(recon_loss_term.item()),
                    "total_variation": "{:.4f}".format(mv_loss_term.item())
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
    plot_losses(metrics, "ae_loss_plots")
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
    torch.save(state, f'models/model_state_ae_{timestamp}.pth')
    print("Best model saved to 'models'.")