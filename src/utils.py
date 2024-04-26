import copy

import torch
from functools import partial
from torch import nn

def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    """
    Linear warm up learning rate scheduler function.
    :param current_step: current step in training process
    :param num_warmup_steps: number of steps to warm up learning rate. That is, number of steps to increase learning rate to 1
    :param num_training_steps: Total number of training steps.
    :return: return dynamic learning rate at current step.
    """
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def collate_fn(data: tuple):
    """
    Collate function that takes a tuple of tensors, pads them with zero to align their dimensions, and returns tensors.
    :param data: tuple of tensors.
    :return: tuple of tensors padded as to have same dimensions.
    """
    src_X, src_fX, tgt_X, tgt_fX = zip(*data)
    src_X  = torch.nn.utils.rnn.pad_sequence(src_X, batch_first=True)
    src_fX = torch.nn.utils.rnn.pad_sequence(src_fX, batch_first=True)
    tgt_X  = torch.nn.utils.rnn.pad_sequence(tgt_X, batch_first=True)
    tgt_fX = torch.nn.utils.rnn.pad_sequence(tgt_fX, batch_first=True)
    return src_X, src_fX, tgt_X, tgt_fX


"""
Early stopping mechanism that signals to stop training if validation loss doesn't improve after a given amount of epochs (patience).
It also makes a deep copy of the model every time the validation loss improves. 
"""
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model: nn.Module | None):
        if val_loss < self.best_loss - self.delta:
            self.counter = 0
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def reset(self):
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False