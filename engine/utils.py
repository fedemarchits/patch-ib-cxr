import torch
import os

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, checkpoint_path='best_model.pt', verbose=True, mode='min'):
        """
        Args:
            patience (int): How many epochs to wait after last time validation metric improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            checkpoint_path (str): Path to save the best model.
            verbose (bool): If True, prints a message for each validation metric improvement.
            mode (str): 'min' for loss (lower is better), 'max' for metrics like recall (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    @property
    def best_loss(self):
        """Backwards compatibility alias for best_score."""
        return self.best_score

    def __call__(self, val_metric, model, optimizer=None, epoch=None):
        """
        Args:
            val_metric: The metric to track (loss if mode='min', recall if mode='max')
        """
        if self.best_score is None:
            self.best_score = val_metric
            self.save_checkpoint(val_metric, model, optimizer, epoch)
        elif self._is_improvement(val_metric):
            self.best_score = val_metric
            self.save_checkpoint(val_metric, model, optimizer, epoch)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_improvement(self, val_metric):
        """Check if the metric improved based on mode."""
        if self.mode == 'min':
            # Lower is better (for loss)
            return val_metric < self.best_score - self.min_delta
        else:
            # Higher is better (for recall, accuracy, etc.)
            return val_metric > self.best_score + self.min_delta

    def save_checkpoint(self, val_metric, model, optimizer=None, epoch=None):
        """Saves model when validation metric improves."""
        if self.verbose:
            direction = "decreased" if self.mode == 'min' else "increased"
            print(f"Validation metric {direction} ({self.best_score:.4f} -> {val_metric:.4f}). Saving model to {self.checkpoint_path}")

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'val_metric': val_metric,
            'val_loss': val_metric if self.mode == 'min' else None,
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch

        torch.save(checkpoint, self.checkpoint_path)