import torch
import numpy as np
import h5py
from collections import OrderedDict

def save_weights_h5(model, filepath):
    """Saves the model's state_dict to an H5 file."""
    with h5py.File(filepath, 'w') as f:
        for name, param in model.state_dict().items():
            f.create_dataset(name, data=param.cpu().numpy())
    print(f"Model weights saved to {filepath}")

def load_weights_h5(model, filepath):
    """Loads weights from an H5 file into the model."""
    with h5py.File(filepath, 'r') as f:
        new_state_dict = OrderedDict()
        for name, dataset in f.items():
            new_state_dict[name] = torch.from_numpy(dataset[()])
    model.load_state_dict(new_state_dict)
    print(f"Model weights loaded from {filepath}")


class EarlyStopping:
    """Early stops the training if validation score doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.h5', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        # --- THIS IS THE CORRECTED LINE ---
        self.val_score_min = -np.inf
        # --- END CORRECTION ---
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_score, model):
        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0

    def save_checkpoint(self, val_score, model):
        '''Saves model when validation score improves.'''
        if self.verbose:
            self.trace_func(f'Validation score improved ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model to {self.path} ...')
        save_weights_h5(model, self.path)
        self.val_score_min = val_score

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}