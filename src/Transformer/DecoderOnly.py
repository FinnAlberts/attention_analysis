import torch
from torch import nn
from torch.nn import functional as F

from src.Transformer import TransformerModules as TB
from src.Transformer.DistanceMetrics import euclidean
import copy

"""
Base class for the autoregressive decoder-only Transformer model. 
Uses naive linear layer for embedding.
"""
class BaseDecoderOnlyTransformer(nn.Module):

    def __init__(self,
                 d_in=2,
                 emb_size=512,
                 n_heads=8,
                 n_hidden=64,
                 ffn_n_hidden=2048,
                 num_layers=3,
                 _attention=TB.DotProductAttention(),
                 norm_first=True):
        """
        :param d_in: total number of input features (N timeseries + M covariates)
        :param emb_size: embedding size of d_in (d_model)
        :param n_heads: number of heads in multi-head attention
        :param n_hidden: number of units in Query, Key, Value projection in multi-head attention
        :param ffn_n_hidden: number of hidden units in point-wise FFN
        :param num_layers: number of attention decoder layers
        :param _attention: attention module (default: DotProductAttention)
        :param norm_first: whether to apply layer normalization before or after attention
        """
        super().__init__()
        self.emb = nn.Linear(d_in, emb_size)
        self.pos_enc = TB.PositionalEncoding(emb_size)

        decoder_block = TB.TransformerEncoderBlock(n_heads=n_heads, n_hidden=n_hidden, n_out=emb_size,
                                                   ffn_n_hidden=ffn_n_hidden, _attention=_attention, norm_first=norm_first)
        self.transformer_blocks = nn.ModuleList(
            [copy.deepcopy(decoder_block) for _ in range(num_layers)]
        )

    def forward(self, X: torch.Tensor, fX: torch.Tensor, mask: torch.Tensor = None):
        # covariates X:  (batch_size, seq_len, cov_d)
        # features fX:   (batch_size, seq_len, feat_d)

        # (batch_size, seq_len, cov_d + feat_d)
        Y = torch.cat([X, fX], dim=-1)

        # embedding: (batch_size, seq_len, cov_d + feat_d) -> (batch_size, seq_len, emb_size)
        # positional embedding: (seq_len) -> (seq_len, emb_size)
        Y = self.pos_enc(self.emb(Y))

        # through decoder blocks with same mask shape (seq_len, seq_len)
        for block in self.transformer_blocks:
            Y = block(Y, mask=mask)

        # output shape: (batch_size, seq_len, emb_size)
        return Y
    
class SimplexDecoderOnlyTransformer(nn.Module):
    def __init__(self,
                 d_in=2,
                 emb_size=512,
                 n_hidden=64,
                 ffn_n_hidden=2048,
                 num_layers=3,
                 norm_first=True,
                 distance_metric=euclidean,
                 conv_out_dim=64,
                 kernel_size=3):
        super().__init__()
        self.emb = nn.Linear(d_in, emb_size)
        self.pos_enc = TB.PositionalEncoding(emb_size)

        decoder_block = TB.SimplexTransformerEncoderBlock(n_out=emb_size, ffn_n_hidden=ffn_n_hidden, norm_first=norm_first,
                                                          distance_metric=distance_metric, conv_out_dim=conv_out_dim, kernel_size=kernel_size)
        self.transformer_blocks = nn.ModuleList(
            [copy.deepcopy(decoder_block) for _ in range(num_layers)]
        )

    def forward(self, X: torch.Tensor, fX: torch.Tensor, mask: torch.Tensor = None):
        # covariates X:  (batch_size, seq_len, cov_d)
        # features fX:   (batch_size, seq_len, feat_d)

        # (batch_size, seq_len, cov_d + feat_d)
        Y = torch.cat([X, fX], dim=-1)

        # embedding: (batch_size, seq_len, cov_d + feat_d) -> (batch_size, seq_len, emb_size)
        # positional embedding: (seq_len) -> (seq_len, emb_size)
        Y = self.pos_enc(self.emb(Y))

        # through decoder blocks with same mask shape (seq_len, seq_len)
        for block in self.transformer_blocks:
            Y = block(Y, mask=mask)

        # output shape: (batch_size, seq_len, emb_size)
        return Y


"""
Decoder-only Transformer model that predicts d_out values for each step.
"""
class PointDecoderOnlyTransformer(BaseDecoderOnlyTransformer):
    def __init__(self,
                 d_in=2,
                 d_out=1,
                 emb_size=512,
                 n_heads=8,
                 n_hidden=64,
                 ffn_n_hidden=2048,
                 num_layers=3,
                 _attention=TB.DotProductAttention(),
                 norm_first=True):
        """
        :param d_in: total number of input features (N timeseries + M covariates)
        :param d_out: number of output features (N timeseries)
        :param emb_size: embedding size of d_in (d_model)
        :param n_heads: number of heads in multi-head attention
        :param n_hidden: number of units in Query, Key, Value projection in multi-head attention
        :param ffn_n_hidden: number of hidden units in point-wise FFN
        :param num_layers: number of attention decoder layers
        :param _attention: attention module (default: DotProductAttention)
        :param norm_first: whether to apply layer normalization before or after attention
        """
        super().__init__(d_in, emb_size, n_heads, n_hidden, ffn_n_hidden, num_layers, _attention, norm_first)
        self.fc = nn.Linear(emb_size, d_out)

    def forward(self, X: torch.Tensor, fX: torch.Tensor, mask: torch.Tensor = None):
        Y = super().forward(X, fX, mask)
        # dense layer to project to d_out dims
        # (batch_size, seq_len, emb_size) -> (batch_size, seq_len, d_out)
        return self.fc(Y)


"""
Decoder-only Transformer model that predicts d_out Gaussian distribution parameters (mu, sigma) for each step. 
These can be used to interpret the autoregressive task probabilistically. 
"""
class GaussianDecoderOnlyTransformer(BaseDecoderOnlyTransformer):
    def __init__(self,
                 d_in=2,
                 d_out=1,
                 emb_size=512,
                 n_heads=8,
                 n_hidden=64,
                 ffn_n_hidden=2048,
                 num_layers=3,
                 _attention=TB.DotProductAttention(),
                 norm_first=True):
        """
        :param d_in: total number of input features (N timeseries + M covariates)
        :param d_out: number of output features (N timeseries)
        :param emb_size: embedding size of d_in (d_model)
        :param n_heads: number of heads in multi-head attention
        :param n_hidden: number of units in Query, Key, Value projection in multi-head attention
        :param ffn_n_hidden: number of hidden units in point-wise FFN
        :param num_layers: number of attention decoder layers
        :param _attention: attention module (default: DotProductAttention)
        :param norm_first: whether to apply layer normalization before or after attention
        """
        super().__init__(d_in, emb_size, n_heads, n_hidden, ffn_n_hidden, num_layers, _attention, norm_first)
        self.lin_mu = nn.Linear(emb_size, d_out)
        self.lin_sigma = nn.Linear(emb_size, d_out)

    def forward(self, X: torch.Tensor, fX: torch.Tensor, mask: torch.Tensor = None):
        Y = super().forward(X, fX, mask)
        # dense layer to project to d_out dims
        # (batch_size, seq_len, emb_size) -> (batch_size, seq_len, d_out)
        return self.lin_mu(Y), F.softplus(self.lin_sigma(Y))


"""
Base class for the autoregressive decoder-only Transformer model. 
Uses convolutional layer for the embedding.
"""
class BaseConvolutionalDecoderOnlyTransformer(nn.Module):

    def __init__(self,
                 d_in=2,
                 emb_size=512,
                 n_heads=8,
                 n_hidden=64,
                 ffn_n_hidden=2048,
                 num_layers=3,
                 norm_first=True,
                 _attention=TB.DotProductAttention(),
                 conv_kernel_size=3,
                 conv_dilation=1
                 ):
        """
        :param d_in: total number of input features (N timeseries + M covariates)
        :param emb_size: embedding size of d_in (d_model)
        :param n_heads: number of heads in multi-head attention
        :param n_hidden: number of units in Query, Key, Value projection in multi-head attention
        :param ffn_n_hidden: number of hidden units in point-wise FFN
        :param num_layers: number of attention decoder layers
        :param norm_first: whether to apply layer normalization before or after attention
        :param _attention: attention module (default: DotProductAttention)
        :param conv_kernel_size: kernel size of the convolutional embedding layer
        :param conv_dilation: dilation of the convolutional embedding layer
        """
        super().__init__()
        self.emb = TB.CausalConv1dEmbedding(d_in, emb_size, conv_kernel_size, conv_dilation)
        self.pos_enc = TB.PositionalEncoding(emb_size)

        decoder_block = TB.TransformerEncoderBlock(n_heads=n_heads, n_hidden=n_hidden, n_out=emb_size,
                                                   ffn_n_hidden=ffn_n_hidden, _attention=_attention, norm_first=norm_first)
        self.transformer_blocks = nn.ModuleList(
            [copy.deepcopy(decoder_block) for _ in range(num_layers)]
        )

    def forward(self, X: torch.Tensor, fX: torch.Tensor, mask: torch.Tensor = None):
        # covariates X:  (batch_size, seq_len, cov_d)
        # features fX:   (batch_size, seq_len, feat_d)

        # (batch_size, seq_len, cov_d + feat_d)
        Y = torch.cat([X, fX], dim=-1)

        Y = Y.transpose(1, 2)

        # embedding: (batch_size, cov_d + feat_d, seq_len) -> (batch_size, emb_size, seq_len)
        Y = self.emb(Y)

        # positional embedding: (seq_len) -> (seq_len, emb_size)
        Y = self.pos_enc(Y.transpose(1, 2))

        # through decoder blocks with same mask shape (seq_len, seq_len)
        for block in self.transformer_blocks:
            Y = block(Y, mask=mask)

        # output shape: (batch_size, seq_len, emb_size)
        return Y


"""
Decoder-only Transformer model with convolutional embedding that predicts d_out values for each step.
"""
class ConvolutionalDecoderOnlyTransformer(BaseConvolutionalDecoderOnlyTransformer):
    def __init__(self,
                 d_in=2,
                 d_out=1,
                 emb_size=512,
                 n_heads=8,
                 n_hidden=64,
                 ffn_n_hidden=2048,
                 num_layers=3,
                 norm_first=True,
                 _attention=TB.DotProductAttention(),
                 conv_kernel_size=3,
                 conv_dilation=1):
        """
        :param d_in: total number of input features (N timeseries + M covariates)
        :param d_out: number of output features (N timeseries)
        :param emb_size: embedding size of d_in (d_model)
        :param n_heads: number of heads in multi-head attention
        :param n_hidden: number of units in Query, Key, Value projection in multi-head attention
        :param ffn_n_hidden: number of hidden units in point-wise FFN
        :param num_layers: number of attention decoder layers
        :param norm_first: whether to apply layer normalization before or after attention
        :param _attention: attention module (default: DotProductAttention)
        :param conv_kernel_size kernel size of the convolutional embedding layer
        :param conv_dilation dilation of the convolutional embedding layer
        """
        super().__init__(d_in, emb_size, n_heads, n_hidden, ffn_n_hidden, num_layers, norm_first,
                         _attention, conv_kernel_size, conv_dilation)
        self.fc = nn.Linear(emb_size, d_out)

    def forward(self, X: torch.Tensor, fX: torch.Tensor, mask: torch.Tensor = None):
        Y = super().forward(X, fX, mask)
        # dense layer to project to d_out dims
        # (batch_size, seq_len, emb_size) -> (batch_size, seq_len, d_out)
        return self.fc(Y)
