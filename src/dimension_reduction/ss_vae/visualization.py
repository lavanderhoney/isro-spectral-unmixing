import matplotlib.pyplot as plt

def plot_losses(metrics, save_path=None):
    phases = ['train', 'val']
    loss_types = ['total', 'recon', 'kl', 'homology']
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()
    
    for i, loss_type in enumerate(loss_types):
        for phase in phases:
            axs[i].plot(metrics.history[phase][loss_type], label=f'{phase} {loss_type}')
        axs[i].set_title(f'{loss_type} loss')
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel('Loss')
        axs[i].legend()
        axs[i].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()