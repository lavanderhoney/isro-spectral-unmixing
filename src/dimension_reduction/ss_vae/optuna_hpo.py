import optuna
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import math
import os
from datetime import datetime

# --- Import components from your project ---
from dimension_reduction.ss_vae.config import get_config
from dimension_reduction.ss_vae.dataloaders import get_dataloaders
from dimension_reduction.ss_vae.spatial_spectral_vae import SpatialSpectralNet
from dimension_reduction.ss_vae.utils import extract_spectral_data

# Suppress tqdm output for cleaner Optuna logs
# You can comment this out if you want to see progress bars for each trial
TQDM_DISABLE = True

class DataParallelProxy(nn.DataParallel):
    """A DataParallel that forwards attribute access to the underlying .module."""
    def __getattr__(self, name):
        try:
            # first try the normal behavior (e.g. .cuda, .forward, .module, .device)
            return super().__getattr__(name)
        except AttributeError:
            # if it's not found on the wrapper, forward to the wrapped module
            return getattr(self.module, name)
        
def objective(trial: optuna.trial.Trial) -> float:
    """
    Defines a single trial for Optuna hyperparameter search.
    A trial consists of:
    1. Sampling hyperparameters.
    2. Building the model and optimizer.
    3. Training and evaluating the model.
    4. Returning the final validation loss.
    """
    # --- 1. Load base config and suggest hyperparameters ---
    config = get_config()

    # Suggest values for the hyperparameters we want to tune
    config.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    config.latent_dim = trial.suggest_int("latent_dim", 16, 128, step=16)
    config.hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    config.beta = trial.suggest_float('beta', 1e-3, 0.1, log=True)
    config.lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
    config.patch_size = trial.suggest_categorical("patch_size", [5, 7, 9, 11])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure dataloaders are created for each trial if parameters like batch_size change
    train_dl, test_dl = get_dataloaders(config.data_path, config.batch_size, config.patch_size)

    model = SpatialSpectralNet(
        train_dl.dataset.__getattribute__('B'),
        config.patch_size,
        config.latent_dim,
        config.hidden_dim,
        config.lstm_layers,
        config.cnn_layers,
        config.free_bits
    ).to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = DataParallelProxy(model)

    # Note: torch.compile can add overhead for short trials.  removing it during hyperparameter search for speed.

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.scheduler_patience)
    recon_loss_fn = nn.MSELoss()

    best_test_loss = float('inf')

    # --- 3. Training and Evaluation Loop ---
    for epoch in range(config.epochs):
        # Training
        model.train()
        train_pbar = tqdm(train_dl, total=len(train_dl), desc=f"Trial {trial.number} Epoch {epoch+1} [Train]", disable=TQDM_DISABLE)
        for x in train_pbar:
            x = x.float().to(device)
            optimizer.zero_grad()
            recon = model(x)
            input_spectra = extract_spectral_data(x)
            recon_loss_term = recon_loss_fn(recon, input_spectra)
            kl_loss = model.encoder.kl_loss_term # type: ignore
            homology_loss = model.encoder.homology_loss_term # type: ignore
            loss = recon_loss_term + config.beta * kl_loss + homology_loss # type: ignore
            
            if math.isnan(loss.item()):
                # If loss is NaN, it's a failed trial. Prune it.
                raise optuna.exceptions.TrialPruned()

            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        total_test_loss = 0
        test_pbar = tqdm(test_dl, total=len(test_dl), desc=f"Trial {trial.number} Epoch {epoch+1} [Eval ]", disable=TQDM_DISABLE)
        with torch.no_grad():
            for x in test_pbar:
                x = x.float().to(device)
                recon = model(x)
                input_spectra = extract_spectral_data(x)
                recon_loss_term = recon_loss_fn(recon, input_spectra)
                kl_loss = model.encoder.kl_loss_term # type: ignore
                homology_loss = model.encoder.homology_loss_term # type: ignore
                loss = recon_loss_term + config.beta * kl_loss + homology_loss # type: ignore
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / len(test_dl)

        # Update the best loss for this trial
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
        
        scheduler.step(avg_test_loss)

        # --- Optuna Pruning ---
        # Report intermediate results to Optuna.
        trial.report(avg_test_loss, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # --- 4. Return the final metric to be optimized ---
    return best_test_loss


if __name__ == "__main__":
    # The Pruner stops unpromising trials early.
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    # Create the study, specifying the direction to "minimize" the loss.
    study = optuna.create_study(direction="minimize", pruner=pruner)

    # Start the optimization. Optuna will call the `objective` function `n_trials` times.
    study.optimize(objective, n_trials=50) # You can change the number of trials

    # --- Print results ---
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print("Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    print("\nBest trial:")
    trial = study.best_trial

    print(f"  Value (min validation loss): {trial.value:.4f}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # You can also save the results of the study
    # df = study.trials_dataframe()
    # df.to_csv("hyperparameter_study_results.csv")