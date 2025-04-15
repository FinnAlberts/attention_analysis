"""
This file contains several buidling blocks for Transformer-based architectures. The following building blocks are implemented:
- Positional encoding module (according to the paper "Attention is all you need" (https://arxiv.org/abs/1706.03762))
- Causal 1D-convolution embedding module
- Scaled dot product attention module
- Multi-head attention module (for scaled dot product attention)
- Transformer encoder block
- Transformer decoder block
"""

import torch
from torch import nn
from torch.nn import functional as F
from src.Transformer.DistanceMetrics import euclidean

import math

"""
Standard positional encoding layer adopted from "Attention Is All You Need"
"""
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

"""
Causal 1-dimensional convolutional embedding.
Might be better than the naive linear projection embedding
"""
class CausalConv1dEmbedding(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, **kwargs):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, **kwargs)

    def forward(self, x):
        return F.leaky_relu(self.conv(F.pad(x, pad=(self.padding, 0))))

"""
Standard Scaled Dot product attention mechanism
"""
class DotProductAttention(nn.Module):

    def __init__(self, dropout_rate: float = 0.1):
        super().__init__()
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

        self.attention_weight = F.softmax(presoftmax, dim=-1)

        # out: (batch_size, n_heads, seq_len, n_hidden/n_heads)
        return self.dropout(self.attention_weight @ vals)


"""
Multi-head attention mechanism.
"""
class MultiHeadAttention(nn.Module):

    def __init__(self, attention: nn.Module, n_heads: int, n_hidden: int, n_out: int, bias=False):
        super().__init__()
        self.n_heads = n_heads
        self.n_hidden = n_hidden
        self.W_q = nn.LazyLinear(n_hidden, bias=bias)
        self.W_k = nn.LazyLinear(n_hidden, bias=bias)
        self.W_v = nn.LazyLinear(n_hidden, bias=bias)
        self.W_o = nn.LazyLinear(n_out)
        self.attention = attention

    def transpose_QKV(self, X: torch.Tensor):
        X = X.reshape(*X.shape[:2], self.n_heads, -1)
        X = X.permute(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, n_hidden/n_heads)
        return X

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor = None):

        K = self.transpose_QKV(self.W_k(keys))
        V = self.transpose_QKV(self.W_v(values))
        Q = self.transpose_QKV(self.W_q(queries))
        # Q, K, V: (batch_size, n_heads, seq_len, n_hidden/n_heads)

        out = self.attention(Q, K, V, mask)
        out = out.reshape(out.shape[0], out.shape[2], -1)  # (batch_size, seq_len, n_hidden*n_heads)

        return self.W_o(out)

"""
A modified attention mechanism which uses the original embedding as the query and key, after applying convolution. The values are also the original embedding, but are shifted by 1.
"""   
class SimplexAttention(nn.Module):
    def __init__(self, distance_metric, conv_out_dim, n_heads=1, kernel_size=3):
        """
        :param distance_metric: distance metric to use for the simplex attention mechanism
        :param conv_out_dim: number of output channels for the convolutional layer
        :param n_heads: number of attention heads
        :param kernel_size: kernel size for the convolutional layer
        """
        super().__init__()

        # We use a 1D convolution to group together a series of consecutive timesteps, which we use for the attention mechanism. 
        self.conv = nn.ModuleList([
                    nn.LazyConv1d(out_channels=conv_out_dim, 
                          kernel_size=kernel_size, 
                          stride=1,  # Always stride 1 to ensure length of sequence is preserved
                          padding=0) # Padding will be done manually to ensure causality
                          for _ in range(n_heads)
                    ])

        self.distance_metric = distance_metric
        self.n_heads = n_heads
        self.conv_out_dim = conv_out_dim

    def forward(self, query: torch.Tensor, keys: torch.Tensor, vals: torch.Tensor, mask: torch.Tensor = None):
        # Q, K, V: (batch_size, seq_len, dim)
        batch_size, seq_len, dim = query.shape
        head_dim = dim // self.n_heads

        # We split the input into n_heads heads, and reshape the input to (batch_size, n_heads, seq_len, head_dim)
        query = query.reshape(batch_size, seq_len, self.n_heads, head_dim).permute(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, head_dim)
        keys = keys.reshape(batch_size, seq_len, self.n_heads, head_dim).permute(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, head_dim)


        query_out = []
        keys_out = []
        for i in range(self.n_heads):
            query_head = query[:, i, :, :].reshape(batch_size, seq_len, head_dim)  # (batch_size, seq_len, head_dim)
            keys_head = keys[:, i, :, :].reshape(batch_size, seq_len, head_dim)  # (batch_size, seq_len, head_dim)

            # We need to permute the dimensions to (batch_size, head_dim, seq_len) for the convolution
            query_head = query_head.permute(0, 2, 1)  # (batch_size, head_dim, seq_len)
            keys_head = keys_head.permute(0, 2, 1)  # (batch_size, head_dim, seq_len)

            # We pad on the left side to ensure that the convolution is causal (i.e. it only looks at past values)
            query_head = F.pad(query_head, (self.conv[i].kernel_size[0] - 1, 0), mode='replicate')
            keys_head = F.pad(keys_head, (self.conv[i].kernel_size[0] - 1, 0), mode='replicate')

            # Apply the convolution to the query and keys
            query_head = self.conv[i](query_head)  # (batch_size, conv_out_dim, seq_len)
            keys_head = self.conv[i](keys_head)  # (batch_size, conv_out_dim, seq_len)

            # We need to permute the dimensions back to (batch_size, seq_len, conv_out_dim)
            query_head = query_head.permute(0, 2, 1)  # (batch_size, seq_len, conv_out_dim)
            keys_head = keys_head.permute(0, 2, 1)  # (batch_size, seq_len, conv_out_dim)

            # Append the query and keys to the list of queries and keys
            query_out.append(query_head)
            keys_out.append(keys_head)

        # We stack the queries and keys to get the final query and keys
        query = torch.stack(query_out, dim=1)  # (batch_size, n_heads, seq_len, conv_out_dim)
        keys = torch.stack(keys_out, dim=1)  # (batch_size, n_heads, seq_len, conv_out_dim)

        # We need to reshape the queries and keys to (batch_size * n_heads, seq_len, conv_out_dim)
        query = query.reshape(batch_size * self.n_heads, seq_len, self.conv_out_dim)  # (batch_size * n_heads, seq_len, conv_out_dim)
        keys = keys.reshape(batch_size * self.n_heads, seq_len, self.conv_out_dim)  # (batch_size * n_heads, seq_len, conv_out_dim)

        # Next for each query, we compute the distance to all keys
        distances = self.distance_metric(query, keys) # (batch_size * n_heads, seq_len, seq_len)

        # We negate the distances, because closer distances should have higher attention weights.
        presoftmax = -distances # (batch_size * n_heads, seq_len, seq_len)

        # Apply the mask to the presoftmax values
        if mask is not None:
            if mask.dim() == 2: # If mask is (seq_len, seq_len), we need to expand it to (batch_size, seq_len, seq_len)
                mask = mask.unsqueeze(0)  # (1, seq_len, seq_len)
                mask = mask.expand(batch_size, -1, -1)  # (batch_size, seq_len, seq_len)

            # mask: (batch_size, seq_len, seq_len), we need to expand it to (batch_size, n_heads, seq_len, seq_len)
            mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
            mask = mask.expand(-1, self.n_heads, -1, -1)  # (batch_size, n_heads, seq_len, seq_len)
            mask = mask.reshape(batch_size * self.n_heads, seq_len, seq_len)

            presoftmax = presoftmax.masked_fill(mask == 0, float('-inf'))

        self.attention_weight = F.softmax(presoftmax, dim=-1) # (batch_size * n_heads, seq_len, seq_len)

        # We shift the attention matrix by 1 to the right, so that the attention is put on the effect of the previous timestep
        # We do this by rolling the attention matrix by 1 to the right
        self.attention_weight = torch.roll(self.attention_weight, shifts=1, dims=-1)  # (batch_size * n_heads, seq_len, seq_len)

        # We also split values into n_heads
        vals = vals.reshape(batch_size, seq_len, self.n_heads, head_dim).permute(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, head_dim)
        vals = vals.reshape(batch_size * self.n_heads, seq_len, head_dim)  # (batch_size * n_heads, seq_len, head_dim)

        # Multiply the attention weights with the shifted values
        out = torch.bmm(self.attention_weight, vals)  # (batch_size * n_heads, seq_len, head_dim)
        out = out.reshape(batch_size, self.n_heads, seq_len, head_dim)  # (batch_size, n_heads, seq_len, head_dim)
        out = out.permute(0, 2, 1, 3)  # (batch_size, seq_len, n_heads, head_dim)
        out = out.reshape(batch_size, seq_len, dim) # (batch_size, seq_len, dim)
        
        return out
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 n_heads=8,
                 n_hidden=64,
                 n_out=512,
                 ffn_n_hidden=2048,
                 _attention=DotProductAttention(),
                 dropout=0.1,
                 norm_first=True):
        """
        :param n_heads: number of attention heads
        :param n_hidden: dimensionality of each attention head
        :param n_out: dimensionality of output (after multi-head attention and after point-wise feedforward network)
        :param ffn_n_hidden: hidden dimension of feedforward network
        :param _attention: self attention module (default: DotProductAttention)
        :param dropout: dropout rate
        :param norm_first: whether to apply layer normalization before attention layer or after
        """
        super().__init__()
        self.norm_first = norm_first
        self.mha = MultiHeadAttention(_attention, n_heads, n_hidden, n_out)
        self.norm1 = nn.LayerNorm(n_out)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.LazyLinear(ffn_n_hidden), nn.ReLU(), nn.LazyLinear(n_out))
        self.norm2 = nn.LayerNorm(n_out)

    def forward(self, X: torch.Tensor, mask: torch.Tensor = None):
        if self.norm_first:
            X = self.norm1(X)
            X = X + self.mha(X, X, X, mask)
            X = X + self.ffn(self.norm2(X))
        else:
            X = self.norm1(X + self.mha(X, X, X, mask))
            X = self.norm2(X + self.ffn(X))

        return self.dropout(X)

"""
A single Transformer decoder block which contains:
1. Multi-head attention layer
2. Point-wise Feedforward network layer
with residual connections and layer normalization
"""
class TransformerDecoderBlock(nn.Module):

    def __init__(self,
                 n_heads=8,
                 n_hidden=64,
                 n_out=512,
                 ffn_n_hidden=2048,
                 _self_attention=DotProductAttention(),
                 _cross_attention=DotProductAttention(),
                 dropout=0.1,
                 norm_first=True):
        """
        :param n_heads: number of attention heads
        :param n_hidden: dimensionality of each attention head
        :param n_out: dimensionality of output (after multi-head attention and after point-wise feedforward network)
        :param ffn_n_hidden: hidden dimension of feedforward network
        :param _self_attention: self attention module (default: DotProductAttention)
        :param _cross_attention: cross attention module (default: DotProductAttention)
        :param dropout: dropout rate
        :param norm_first: whether to apply layer normalization before attention layer or after
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first
        self.mha1 = MultiHeadAttention(_self_attention, n_heads, n_hidden, n_out)
        self.norm1 = nn.LayerNorm(n_out)

        self.mha2 = MultiHeadAttention(_cross_attention, n_heads, n_hidden, n_out)
        self.norm2 = nn.LayerNorm(n_out)

        self.ffn = nn.Sequential(nn.LazyLinear(ffn_n_hidden), nn.ReLU(), nn.LazyLinear(n_out))
        self.norm3 = nn.LayerNorm(n_out)

    def forward(self, X: torch.Tensor, enc_outputs: torch.Tensor, mask: torch.Tensor = None):
        # X: (batch_size, seq_len, emb)
        # enc_outputs: (batch_size, seq_len, emb)
        # mask: (seq_len, seq_len)
        if self.norm_first:
            X = self.norm1(X)
            X = X + self.mha1(X, X, X, mask)
            X = self.norm2(X)
            X = X + self.mha2(X, enc_outputs, enc_outputs)
            X = X + self.ffn(self.norm3(X))
        else:
            X = self.norm1(X + self.mha1(X, X, X, mask))
            X = self.norm2(X + self.mha2(X, enc_outputs, enc_outputs))
            X = self.norm3(X + self.ffn(X))

        return self.dropout(X)

"""
A modified Transformer encoder block which uses the Simplex attention mechanism.
"""
class SimplexTransformerEncoderBlock(nn.Module):
    def __init__(self,
                 n_out=512,
                 n_heads=1,
                 ffn_n_hidden=2048,
                 dropout=0.1,
                 norm_first=True,
                 distance_metric=euclidean,
                 conv_out_dim=64,
                 kernel_size=3):
        """
        :param n_out: dimensionality of output
        :param n_heads: number of attention heads
        :param ffn_n_hidden: hidden dimension of feedforward network
        :param dropout: dropout rate
        :param norm_first: whether to apply layer normalization before attention layer or after
        :param distance_metric: distance metric to use for the simplex attention mechanism
        :param conv_out_dim: number of output channels for the convolutional layer
        :param kernel_size: kernel size for the convolutional layer
        """
        super().__init__()
        self.norm_first = norm_first
        self.attention = SimplexAttention(distance_metric, conv_out_dim=conv_out_dim, n_heads=n_heads, kernel_size=kernel_size)
        self.norm1 = nn.LayerNorm(n_out)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.LazyLinear(ffn_n_hidden), nn.ReLU(), nn.LazyLinear(n_out))
        self.norm2 = nn.LayerNorm(n_out)

    def forward(self, X: torch.Tensor, mask: torch.Tensor = None):
        if self.norm_first:
            X = self.norm1(X)
            X = X + self.attention(X, X, X, mask)
            X = X + self.ffn(self.norm2(X))
        else:
            X = self.norm1(X + self.attention(X, X, X, mask))
            X = self.norm2(X + self.ffn(X))

        return self.dropout(X)