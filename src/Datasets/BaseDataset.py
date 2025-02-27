import torch
from torch.utils.data import Dataset
from typing import Literal

class BaseTimeSeriesDataset(Dataset):
    def __init__(self,
                 X: torch.Tensor,
                 fX: torch.Tensor,
                 seq_len: int,
                 shift: int,
                 mode: Literal['overlapping', 'nonoverlapping'] = 'overlapping'):
        """
        :param X: time steps (covariates)
        :param fX: features per time step
        :param seq_len: length of each sequence example in dataset
        :param shift: number of steps to shift the target values
        :param mode: control type of samples.
        - 'overlapping' means that a sliding window samples each sequence where the source and target overlap
        - 'nonoverlapping' means that a sliding window samples each sequence where the source and target do not overlap with each other
        """
        self.seq_len = seq_len
        self.shift = shift
        self.mode = mode
        self.X = X
        self.fX = fX
        self.mask = torch.tril(torch.ones((seq_len, seq_len)), diagonal=0)

    def __getitem__(self, index):
        overlap_shift = None
        if self.mode == 'nonoverlapping':
            overlap_shift = self.seq_len
        elif self.mode == 'overlapping':
            overlap_shift = self.shift

        sample = (
            self.X[index: index + self.seq_len],
            self.fX[index: index + self.seq_len],
            self.X[index + overlap_shift: index + self.seq_len + self.shift],
            self.fX[index + overlap_shift: index + self.seq_len + self.shift],
        )
        return sample

    def __len__(self):
        return len(self.X) - self.seq_len - self.shift