import torch
from torch import nn
from torch.nn import functional as F


"""
Apply a shift operation on the attention weights during inference.
"""
class ShiftAttention(nn.Module):

    def __init__(self, shift: int = 0, dropout_rate: float = 0.1):
        super().__init__()
        self.shift = shift
        self.dropout = nn.Dropout(dropout_rate)
        self.attention_weight = None

    def forward(self, query: torch.Tensor, keys: torch.Tensor, vals: torch.Tensor, mask: torch.Tensor = None):
        # query/keys: (batch_size, n_heads, seq_len, n_hidden/n_heads)
        # vals:       (batch_size, n_heads, seq_len, n_hidden/n_heads)
        _, _, seq_len, d = query.shape

        # (batch_size, n_heads, seq_len, seq_len)
        presoftmax = query @ keys.transpose(-2, -1) / d ** 0.5
        if mask is not None:
            presoftmax = presoftmax.masked_fill(mask == 0, float('-inf'))

        # Shift rows of the attention matrix to the right and pad with zeros
        if not self.training and self.shift > 0:
            presoftmax = torch.roll(presoftmax, shifts=self.shift, dims=-1)
            presoftmax[..., :self.shift] = float('-inf')

        self.attention_weight = F.softmax(presoftmax, dim=-1)

        # out: (batch_size, n_heads, seq_len, n_hidden/n_heads)
        return self.dropout(self.attention_weight @ vals)
