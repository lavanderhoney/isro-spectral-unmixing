from time import sleep
from spatial_spectral_vae import SpatialSpectralNet
from dataloaders import get_dataloaders, spectral_bands, neighborhood_size
from utils import extract_spectral_data
from tqdm import tqdm
import math
import torch
import torch.optim as optim
import torch.nn as nn
from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler
)
#%%
#--------- TO-DO -----------
# implement parser thing, which takes parameters input from user in CLI, just like in the dade wood's repo
# change the dataloaders accordingly, i.e, it accepts the data path, batch size, patch size from the user
#---------------------------
#%%
batch_size=32
n_bands = spectral_bands
s = neighborhood_size
latent_dimension = 12
hidden_dim = 64
lstm_layers=3
cnn_layers=3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dl, test_dl = get_dataloaders(batch_size)

print("Dataloaders created !")
#%%
model = SpatialSpectralNet(
    n_bands,
    s,
    latent_dimension,
    hidden_dim
).to(device)

print("The Training Begins !")
sleep(0.5)

n_epochs = 30
update_iters = 1000
scheduler_patience=3
early_stop_patience=5
optimizer = optim.Adam(model.parameters(), lr=1e-4, )
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience, threshold=0.01)
patience_cntr = 0
best_model_state = None
# best_test_loss = float('inf')
best_test_loss = float('inf')
recon_loss = nn.MSELoss()

train_loss = []
train_recon_loss = []
train_kl_loss = []
train_homology_loss = []
test_loss = []
test_recon_loss = []
test_kl_loss = []
test_homology_loss = []

# 2) Set up profiler
profiler = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler("./profiler_logs"),
    record_shapes=True,
    with_stack=True
)

for epoch in range(n_epochs):
    
    # TRAINING
    model.train()
    epoch_train_loss = 0.0
    epoch_train_recon_loss = 0.0
    epoch_train_kl_loss = 0.0
    epoch_train_homology_loss = 0.0
        
    beta = 0.01
    train_pbar = tqdm(train_dl, total=len(train_dl), desc=f"Epoch {epoch+1}/{n_epochs} [Train]")

    for i, x in enumerate(train_pbar):
        x = x.float().to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        model.train()
        recon = model(x)
        input_spectra = extract_spectral_data(x)
        recon_loss_term = recon_loss(recon, input_spectra)
        
        kl_loss = model.encoder.kl_loss_term
        homology_loss = model.encoder.homology_loss_term
        
        loss = recon_loss_term + beta*kl_loss + homology_loss #beta-VAE
        
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        epoch_train_recon_loss += recon_loss_term.item()
        epoch_train_kl_loss += kl_loss.item()
        epoch_train_homology_loss += homology_loss.item()
        
        if i % update_iters == 0:
            train_pbar.set_postfix({
                "loss": "{:.4f}".format(loss.item()),
                "reconstruction": "{:.4f}".format(recon_loss_term.item()),
                "kl": "{:.4f}".format(kl_loss),
                "homology": "{:.4f}".format(homology_loss),
            })
            if math.isnan(loss.item()):
                raise ValueError("Loss went to nan.")
        profiler.step() 
    profiler.stop()
    avg_train_loss = epoch_train_loss / len(train_dl)
    avg_train_recon_loss = epoch_train_recon_loss / len(train_dl)
    avg_train_kl_loss = epoch_train_kl_loss / len(train_dl)
    avg_train_homology_loss = epoch_train_homology_loss / len(train_dl)
    
    train_loss.append(avg_train_loss)
    train_recon_loss.append(avg_train_recon_loss)
    train_kl_loss.append(avg_train_kl_loss)
    train_homology_loss.append(avg_train_homology_loss)
    
    # EVAL
    model.eval()
    epoch_test_loss = 0.0
    epoch_test_recon_loss = 0.0
    epoch_test_kl_loss = 0.0
    epoch_test_homology_loss = 0.0
    test_pbar = tqdm(test_dl, total=len(test_dl), desc=f"Epoch {epoch+1}/{n_epochs} [Eval ]")
    for x in test_pbar:
        x=x.float().to(device, non_blocking=True)
        with torch.inference_mode():
            recon = model(x)
            input_spectra = extract_spectral_data(x)
            recon_loss_term = recon_loss(recon, input_spectra)

            kl_loss = model.encoder.kl_loss_term
            homology_loss = model.encoder.homology_loss_term

            loss = recon_loss_term + beta*kl_loss + homology_loss #beta-VAE
            epoch_test_loss += loss.item()
            epoch_test_recon_loss += recon_loss_term.item()
            epoch_test_kl_loss += kl_loss.item()
            epoch_test_homology_loss += homology_loss.item()
        test_pbar.set_postfix({
                "loss": "{:.4f}".format(loss.item()),
                "reconstruction": "{:.4f}".format(recon_loss_term.item()),
                "kl": "{:.4f}".format(kl_loss),
                "homology": "{:.4f}".format(homology_loss),
            })
            
    avg_test_loss = epoch_test_loss / len(test_dl)
    avg_test_recon_loss = epoch_test_recon_loss / len(test_dl)
    avg_test_kl_loss = epoch_test_kl_loss / len(test_dl) 
    avg_test_homology_loss = epoch_test_homology_loss / len(test_dl)   
    test_loss.append(avg_test_loss)
    test_recon_loss.append(avg_test_recon_loss)
    test_kl_loss.append(avg_test_kl_loss)
    test_homology_loss.append(avg_test_homology_loss)   
    
    scheduler.step(avg_test_loss)
    
    if avg_test_loss < best_test_loss:
        print("New best model found")
        best_test_loss = avg_test_loss
        best_model_state = model.state_dict()
        patience_cntr = 0
    else:
        patience_cntr += 1
    if patience_cntr >= early_stop_patience:
        print(f"Early stopping triggered at epoch: {epoch}")
        break

# %%
#plotting
from matplotlib import pyplot as plt
# Plot total losses
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Train Total Loss')
plt.plot(test_loss, label='Test Total Loss')
plt.title("Total Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Plot reconstruction losses
plt.figure(figsize=(10, 6))
plt.plot(train_recon_loss, label='Train Reconstruction Loss')
plt.plot(test_recon_loss, label='Test Reconstruction Loss')
plt.title("Reconstruction Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Plot KL divergence losses
plt.figure(figsize=(10, 6))
plt.plot(train_kl_loss, label='Train KL Divergence Loss')
plt.plot(test_kl_loss, label='Test KL Divergence Loss')
plt.title("KL Divergence Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

#Plot Homology losses
plt.figure(figsize=(10, 6))
plt.plot(train_homology_loss, label='Train Homology Loss')
plt.plot(test_homology_loss, label='Test Homology Loss')
plt.title("Homology Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()