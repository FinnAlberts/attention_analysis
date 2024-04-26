import torch
from torch import nn
from torch.nn import functional as F


"""
Apply convolution on the attention weights of a canonical scaled dot-product attention mechanism.
 
"""
class MultiFilterConvDotProductAttention(nn.Module):
    def __init__(self, dropout_rate: float = 0.1, kernel_size: int = 3, num_filters: int = 4):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.attention_weight = None
        self.num_filters = num_filters
        # Multi-filter convolution layer
        self.conv = nn.Conv2d(in_channels=1, out_channels=num_filters,
                              kernel_size=kernel_size, padding=kernel_size // 2)
        self.linear = nn.Linear(num_filters, 1)
        self.scale = None

    def forward(self, query: torch.Tensor, keys: torch.Tensor, vals: torch.Tensor, mask: torch.Tensor = None):
        _, _, seq_len, d = query.shape
        self.scale = d ** 0.5  # Scale factor

        # Compute scaled dot-product attention
        dot_product = query @ keys.transpose(-2, -1) / self.scale

        # Reshape attention weights for convolutional processing:
        # (batch_size * n_heads, 1, seq_len, seq_len)
        batch_size, n_heads, seq_len, _ = dot_product.shape
        reshaped_weights = dot_product.view(batch_size * n_heads, 1, seq_len, seq_len)

        # Apply convolution to the reshaped attention weights
        conv_weights = F.leaky_relu(self.conv(reshaped_weights))

        # Reshape to mix down the filter dimension:
        # (batch_size, n_heads, num_filters, seq_len, seq_len)
        conv_weights = conv_weights.view(batch_size, n_heads, self.num_filters, seq_len, seq_len)

        # Desired shape: (batch_size, n_heads, seq_len, seq_len, num_filters)
        conv_weights = conv_weights.permute(0, 1, 3, 4, 2)

        preattn = self.linear(conv_weights).squeeze(dim=-1)

        if mask is not None:
            preattn = preattn.masked_fill(mask == 0, float('-inf'))

        self.attention_weight = F.softmax(preattn, dim=-1)

        # Apply dropout to the combined weights and multiply with values
        output = self.dropout(self.attention_weight) @ vals
        return output
