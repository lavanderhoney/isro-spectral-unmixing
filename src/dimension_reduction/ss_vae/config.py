import argparse

def get_config()-> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Spatial-Spectral VAE Training')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='/teamspace/studios/this_studio/isro-spectral-unmixing/data/den_reflectance_ch2_iir_nci_20191208T0814159609_d_img_d18.npz', help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size')
    parser.add_argument('--patch_size', type=int, default=5, help='Spatial neighborhood size')
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=12, help='Latent space dimension')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--lstm_layers', type=int, default=3, help='Number of LSTM layers')
    parser.add_argument('--cnn_layers', type=int, default=3, help='Number of CNN layers')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--beta', type=float, default=1, help='KL loss weight')
    parser.add_argument('--free_bits', type=float, default=0.1, help='Free bits for KL divergence')
    parser.add_argument('--update_interval', type=int, default=1000, help='Log interval in iterations')
    
    # Utility parameters
    parser.add_argument('--early_stop', type=int, default=5, help='Early stopping patience')
    parser.add_argument("--scheduler_patience", type=int, default=3, help='Scheduler patience for learning rate reduction')
    
    return parser.parse_args()