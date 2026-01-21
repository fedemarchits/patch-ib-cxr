import torch
import os

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, checkpoint_path='best_model.pt', verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            checkpoint_path (str): Path to save the best model.
            verbose (bool): If True, prints a message for each validation loss improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model, optimizer=None, epoch=None):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer, epoch)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer=None, epoch=None):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f"Validation loss decreased ({self.best_loss:.4f}). Saving model to {self.checkpoint_path}")

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch

        torch.save(checkpoint, self.checkpoint_path)