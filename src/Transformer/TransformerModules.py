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
