import numpy as np
import torch
import math
from time import sleep
from dimension_reduction.ss_vae.config import get_config
from dimension_reduction.ss_vae.metrics_logger import MetricsLogger
from dimension_reduction.ss_vae.visualization import plot_losses
from dimension_reduction.ss_vae.dataloaders import get_dataloaders, open_datacube
from mineral_analysis.unmixing.ae import AE, spectral_angle_distance_loss, total_variation, spectral_information_divergence_loss, entropy_loss
from mineral_analysis.endmember_extraction import extract_endmembers
from tqdm import tqdm


def main():
    """
    Main function to train the LMM based Autoencoder (AE) on reflectance data with two stages.
    """
    torch.autograd.set_detect_anomaly(True)

    # Load and preprocess the reflectance data
    refl_cube_path = '/teamspace/studios/this_studio/isro-spectral-unmixing/data/den_reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz' #the denoised image
    train_dl, test_dl, wavelengths = get_dataloaders(refl_cube_path)
    H, wavelengths = open_datacube(refl_cube_path)
    H_t = H.transpose(1, 2, 0)  #(H, W, bands)
    # Get a batch to determine the number of spectral bands
    first_batch = next(iter(train_dl))
    n_bands = first_batch[0].shape[1]  # Number of spectral bands
    config = get_config()
    metrics = MetricsLogger()
    
    # Initialize the decoder matrix E as endmembers from VCA
    ems, _ = extract_endmembers(H_t, wavelengths, algorithm='vca', n_endmembers=4, show_endmembers=False, show_amaps=False)
    print("VCA done")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE(
        input_dim=n_bands,  # Number of bands
        hidden_dim=config.hidden_dim,
        latent_dim=4,
        em_spectra=ems,
        constraint='man',  
        scaling=1
    ).to(device)
    # model = torch.compile(model)

    epoch_stage = 2 # train decoder after these epochs
    
    # Freeze decoder (E) initially. Better to keep it frozen until stage 2, if not, it accumaltes gradients which are not used. Also much cleaner code
    model.E.requires_grad_(False)
    
    encoder_params = [p for p in model.parameters() if p is not model.E and p.requires_grad]
    optim_encoder = torch.optim.Adam(encoder_params, lr=config.lr)
    scheduler_encoder = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_encoder, patience=config.scheduler_patience, threshold=0.01)
    
    optim_decoder = None # will create later in the training loop
    scheduler_decoder = None
    
    patience_cntr = 0
    best_model_state = None
    best_test_loss = float('inf')
   
    print("The two stage Training Begins !")
    sleep(0.5)

    w_recon, w_mv, w_sid, w_ent = 1, 1, 1, 0 # Weights for the losses
    for epoch in range(config.epochs):
        metrics.reset_epoch()

        # TRAINING
        model.train()
        train_pbar = tqdm(train_dl, total=len(train_dl), desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        for i, x in enumerate(train_pbar):
            x = x[0].float().to(device)  # Extract the tensor from the tuple
                        
            # unfreeze and init the decoder only once (not checking with > condn)
            if epoch == epoch_stage+1 and not model.E.requires_grad:
                model.E.requires_grad_(True)
                decoder_params = [model.E] # since this is the only param in decoder for now (LMM)
                optim_decoder = torch.optim.Adam(decoder_params, lr=config.lr)
                scheduler_decoder = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_decoder, patience=config.scheduler_patience, threshold=0.01)
                print("Decoder unfrozen and optimizer created.")
            
            optim_encoder.zero_grad()
            if optim_decoder is not None:
                optim_decoder.zero_grad()
                
            try:
                x_hat, z, E_positive = model(x)
            except Exception as e:
                print("Error caught! ")
                print(e)
                for name, param in model.named_parameters():
                    if "z" in name or "E_positive" in name:
                        print(f"{name}: {param.shape} - {param}")
            assert not torch.isnan(x_hat).any(), "NaN in x_hat"
            assert not torch.isnan(x).any(), "NaN in x"
            
            recon_loss_term = spectral_angle_distance_loss(x_hat, x)
            mv_loss_term = total_variation(E_positive)
            sid_loss_term = spectral_information_divergence_loss(x_hat, x)
            entropy_loss_term = entropy_loss(z)
            
            loss = w_recon*recon_loss_term + w_mv* mv_loss_term + w_sid*sid_loss_term - w_ent*entropy_loss_term
            # assert not torch.isnan(loss).any(), "NaN in loss"
            
            if not torch.isfinite(loss):
                print("DEBUG: loss is not finite:", loss)
                # print breakdown of losses if you compute components:
                print("reconstruction:", recon_loss_term.item(), "sid:", sid_loss_term.item(), "tv:", mv_loss_term.item(), "entropy:", entropy_loss_term.item())
                # optionally save a small batch for inspection
                torch.save({
                    'x': x.detach().cpu(),
                    'x_hat': x_hat.detach().cpu(),
                    'z': z.detach().cpu(),
                    'E_positive': E_positive.detach().cpu()
                }, "debug_nan_batch.pt")
                raise RuntimeError("Non-finite loss")

            loss.backward()
            # for name, param in model.named_parameters():
            #     if param is not None and torch.isnan(param).any() or torch.isinf(param).any() or torch.isnan(param.grad).any(): # type: ignore
            #         print(f"============={name}: {param.shape} - {param.grad}=============")
            #         print(f"Parameter {name} has NaN values.")
                    
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping to avoid exploding gradients
            
            optim_encoder.step()
            if optim_decoder is not None:
                optim_decoder.step() # type: ignore

            metrics.update(phase='train', total=loss.item(), recon=recon_loss_term.item(), mv=mv_loss_term.item(), sid=sid_loss_term.item(), homology=entropy_loss_term.item()) # store entropy in metrics under homology as surrogate

            train_pbar.set_postfix({
                "loss": "{:.5f}".format(loss.item()),
                "reconstruction": "{:.4f}".format(recon_loss_term.item()),
                "total_variation": "{:.4f}".format(mv_loss_term.item()),
                "sid_loss": "{:.4f}".format(sid_loss_term.item()),
                "entropy": "{:.4f}".format(entropy_loss_term.item())
            })
            if math.isnan(loss.item()):
                print(E_positive)
                print(z)
                print(x_hat)
                # for name, param in model.named_parameters():
                #     if torch.isnan(param).any():
                #         print(f"Parameter {name} has NaN values.")
                
                # print("Param gradients:")
                # for name, param in model.named_parameters():
                #     print(f"{name}: {param.shape} - {param.grad}")
                raise ValueError("Loss went to nan.")
            avg_train_loss = metrics.finalize_epoch('train')
                
        # EVALUATION
        model.eval()
        test_pbar = tqdm(test_dl, total=len(test_dl), desc=f"Epoch {epoch+1}/{config.epochs} [Test]")
        with torch.no_grad():
            for x in test_pbar:
                x = x[0].float().to(device)

                x_hat, z, E_positive = model(x)
                
                recon_loss_term = spectral_angle_distance_loss(x_hat, x)
                mv_loss_term = total_variation(E_positive)
                sid_loss_term = spectral_information_divergence_loss(x_hat, x)
                entropy_loss_term = entropy_loss(z)
                
                loss = w_recon*recon_loss_term + w_mv* mv_loss_term + w_sid*sid_loss_term + w_ent*entropy_loss_term

                metrics.update(phase='val', total=loss.item(), recon=recon_loss_term.item(), mv=mv_loss_term.item(), sid=sid_loss_term.item(), homology=entropy_loss_term.item())  # store entropy in metrics under homology as surrogate
                test_pbar.set_postfix({
                    "loss": "{:.5f}".format(loss.item()),
                    "reconstruction": "{:.4f}".format(recon_loss_term.item()),
                    "total_variation": "{:.4f}".format(mv_loss_term.item()),
                    "sid_loss": "{:.4f}".format(sid_loss_term.item()),
                    "entropy": "{:.4f}".format(entropy_loss_term.item())
                })


        avg_test_loss = metrics.finalize_epoch('val')
        scheduler_encoder.step(avg_test_loss)
        
        if scheduler_decoder is not None and optim_decoder is not None:
            scheduler_decoder.step(avg_test_loss)

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            print("New best model found.")
            best_model_state = model.state_dict()
            patience_cntr = 0
        else:
            patience_cntr += 1

        if patience_cntr >= config.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    # plot_losses(metrics, "ae_loss_plots")
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
        'config': get_config(), #change this to save actual model params
        'timestamp': timestamp
    }
    torch.save(state, f'models/model_state_ae_stage_{timestamp}.pth')
    print("Best model saved to 'models'.")
    # plot_losses(metrics, "ae_loss_plots")