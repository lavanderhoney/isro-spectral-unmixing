import numpy as np
from typing import Literal, Optional, Union
class MetricsLogger:
    def __init__(self) -> None:
        """
        Initializes the MetricsLogger.
        This class is used to log and track the training and validation metrics during the training process.
        It maintains a history of metrics for both training and validation phases, including total loss, reconstruction loss,
        KL divergence loss, and homology loss.
        This history can be accessed by `self.history`
        """
        self.reset_epoch()
        self.history = {
            'train': {'total': [], 'recon': [], 'kl': [], 'homology': []},
            'val': {'total': [], 'recon': [], 'kl': [], 'homology': []}
        }
    
    def reset_epoch(self) -> None:
        """
        Resets the metrics for a new epoch.
        This method initializes the metrics for both training and validation phases.
        It should be called at the start of each epoch to clear previous metrics.
        """
        self.epoch_metrics = {
            'train': {'total': 0.0, 'recon': 0.0, 'kl': 0.0, 'homology': 0.0, 'count': 0},
            'val': {'total': 0.0, 'recon': 0.0, 'kl': 0.0, 'homology': 0.0, 'count': 0}
        }
    
    def update(self, phase: Literal['train', 'val'], total: float, recon: float, kl: float, homology: Optional[float]=None) -> None:
        """
        Updates the metrics for the current epoch, in the `self.epoch_metrics` dict.
        This method accumulates the losses for the specified phase ('train' or 'val').
        It should be called after a forward pass in the training and validation loops to log the losses for each batch.
        Args:
            phase (str): 'train' or 'val' indicating the phase of training.
            total (float): Total loss.
            recon (float): Reconstruction loss.
            kl (float): KL divergence loss.
            homology (float): Homology loss.
        """
        metrics = self.epoch_metrics[phase]
        metrics['total'] += total
        metrics['recon'] += recon
        metrics['kl'] += kl
        if homology:
            metrics['homology'] += homology
        metrics['count'] += 1
    
    def finalize_epoch(self, phase: Literal['train', 'val']) -> float:
        """
        Finalizes the metrics for the current epoch and appends them to the history dict.
        This method calculates the average losses for the specified phase ('train' or 'val') and appends them to the history, in the `self.history` dict.
        It should be called at the end of each epoch to finalize the metrics and prepare for the next epoch.
        Args:
            phase (str): 'train' or 'val' indicating the phase of training.
        Returns:
            float: The average total loss for the phase."""
        metrics = self.epoch_metrics[phase]
        count = metrics['count']
        self.history[phase]['total'].append(metrics['total'] / count)
        self.history[phase]['recon'].append(metrics['recon'] / count)
        self.history[phase]['kl'].append(metrics['kl'] / count)
        if 'homology' in metrics:
            self.history[phase]['homology'].append(metrics['homology'] / count)
        return self.history[phase]['total'][-1]
    
    def get_latest(self, phase):
        return {k: v[-1] for k, v in self.history[phase].items()}
    
    def save_history(self, path):
        np.save(path, np.array(self.history))