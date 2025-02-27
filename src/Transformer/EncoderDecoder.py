import torch
from torch import nn
from torch.nn import functional as F

from src.Transformer import TransformerModules as TB
import copy

"""
Base class for the encoder-decoder canonical Transformer model. 
Uses naive linear layer for embedding.
"""
class BaseEncoderDecoderTransformer(nn.Module):

    def __init__(self,
                 enc_d_in=2,
                 dec_d_in=2,
                 emb_size=512,
                 n_heads=8,
                 n_hidden=64,
                 ffn_n_hidden=2048,
                 num_layers=3,
                 norm_first=True):
        """
        :param enc_d_in: total number of encoder input features (N timeseries + M covariates)
        :param dec_d_in: total number of decoder input features (N timeseries + M covariates)
        :param emb_size: embedding size of enc_d_in and dec_d_in (i.e., d_model)
        :param n_heads: number of heads in multi-head attention
        :param n_hidden: number of units in Query, Key, Value projection in multi-head attention
        :param ffn_n_hidden: number of hidden units in point-wise FFN
        :param num_layers: number of attention decoder layers
        :param norm_first: whether to apply layer normalization before or after attention
        """

        super().__init__()
        self.enc_emb = nn.Linear(enc_d_in, emb_size)
        self.dec_emb = nn.Linear(dec_d_in, emb_size)

        # learnable
        # self.pos_enc = nn.Embedding(seq_len, emb_size)
        self.pos_enc = TB.PositionalEncoding(emb_size)

        # encoder block
        enc_block = TB.TransformerEncoderBlock(n_heads=n_heads, n_hidden=n_hidden, n_out=emb_size,
                                            ffn_n_hidden=ffn_n_hidden, norm_first=norm_first)

        # decoder block
        dec_block = TB.TransformerDecoderBlock(n_heads=n_heads, n_hidden=n_hidden, n_out=emb_size,
                                           ffn_n_hidden=ffn_n_hidden, norm_first=norm_first)

        # make num_layers clones of encoder block
        self.encoder_blocks = nn.ModuleList(
            [copy.deepcopy(enc_block) for _ in range(num_layers)]
        )
        # make num_layers clones of decoder block
        self.decoder_blocks = nn.ModuleList(
            [copy.deepcopy(dec_block) for _ in range(num_layers)]
        )

        # attention matrices
        self.enc_attn_weights = None
        self.dec_attn_weights = None

    def forward(self, src_X: torch.Tensor, src_fX: torch.Tensor, tgt_X: torch.Tensor, tgt_fX: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass of the encoder-decoder Transformer.
        :param src_X: covariates of source time series with shape (batch_size, seq_len, cov_d)
        :param src_fX: features of source time series with shape (batch_size, seq_len, feat_d)
        :param tgt_X: covariates of target time series with shape (batch_size, seq_len, cov_d)
        :param tgt_fX: features of target time series with shape (batch_size, seq_len, feat_d)
        :param mask: attention mask used for autoregressive process with shape (seq_len, seq_len)
        :return: output with shape (batch_size, seq_len, emb_size)
        """

        # concatenate (batch_size, seq_len, cov_d + feat_d)
        Y = torch.cat([src_X, src_fX], dim=-1)
        # encoder embedding: (batch_size, seq_len, cov_d + feat_d) -> (batch_size, seq_len, emb_size)
        Y = self.pos_enc(self.enc_emb(Y))

        # concatenate (batch_size, seq_len, cov_d + feat_d)
        Z = torch.cat([tgt_X, tgt_fX], dim=-1)
        # decoder embedding: (batch_size, seq_len, feat_d) -> (batch_size, seq_len, emb_size)
        Z = self.pos_enc(self.dec_emb(Z))

        # reset encoder attention weights
        self.enc_attn_weights = [None] * len(self.encoder_blocks)

        # pass through encoder blocks
        for i, block in enumerate(self.encoder_blocks):
            Y = block(Y)
            self.enc_attn_weights[i] = block.mha.attention.attention_weight

        # reset decoder attention weights
        self.dec_attn_weights = [[None] * len(self.decoder_blocks) for _ in range(2)]
        # pass through decoder blocks with same mask shape (seq_len, seq_len)
        for i, block in enumerate(self.decoder_blocks):
            Z = block(Z, Y, mask)
            self.dec_attn_weights[0][i] = block.mha1.attention.attention_weight # self-attention
            self.dec_attn_weights[1][i] = block.mha2.attention.attention_weight # cross-attention

        return Z

class PointEncoderDecoderTransformer(BaseEncoderDecoderTransformer):

    def __init__(self,
                 enc_d_in=2,
                 dec_d_in=1,
                 d_out=1,
                 emb_size=512,
                 n_heads=8,
                 n_hidden=64,
                 ffn_n_hidden=2048,
                 num_layers=3,
                 norm_first=True):
        """
        :param enc_d_in: total number of encoder input features (N timeseries + M covariates)
        :param dec_d_in: total number of decoder input features (N timeseries + M covariates)
        :param d_out: total number of decoder output features
        :param emb_size: embedding size of enc_d_in and dec_d_in (i.e., d_model)
        :param n_heads: number of heads in multi-head attention
        :param n_hidden: number of units in Query, Key, Value projection in multi-head attention
        :param ffn_n_hidden: number of hidden units in point-wise FFN
        :param num_layers: number of attention decoder layers
        :param norm_first: whether to apply layer normalization before or after attention
        """
        super().__init__(enc_d_in, dec_d_in, emb_size, n_heads, n_hidden, ffn_n_hidden, num_layers, norm_first)
        self.fc = nn.Linear(emb_size, d_out)

    def forward(self, src_X: torch.Tensor, src_fX: torch.Tensor, tgt_X: torch.Tensor, tgt_fX: torch.Tensor, mask: torch.Tensor = None):
        Z = super().forward(src_X, src_fX, tgt_X, tgt_fX, mask)
        # dense layer to project to d_out dims
        # (batch_size, seq_len, emb_size) -> (batch_size, seq_len, d_out)
        return self.fc(Z)

class GaussianEncoderDecoderTransformer(BaseEncoderDecoderTransformer):

    def __init__(self,
                 enc_d_in=2,
                 dec_d_in=1,
                 d_out=1,
                 emb_size=512,
                 n_heads=8,
                 n_hidden=64,
                 ffn_n_hidden=2048,
                 num_layers=3,
                 norm_first=True):
        """
        :param enc_d_in: total number of encoder input features (N timeseries + M covariates)
        :param dec_d_in: total number of decoder input features (N timeseries + M covariates)
        :param d_out: total number of decoder output features
        :param emb_size: embedding size of enc_d_in and dec_d_in (i.e., d_model)
        :param n_heads: number of heads in multi-head attention
        :param n_hidden: number of units in Query, Key, Value projection in multi-head attention
        :param ffn_n_hidden: number of hidden units in point-wise FFN
        :param num_layers: number of attention decoder layers
        :param norm_first: whether to apply layer normalization before or after attention
        """
        super().__init__(enc_d_in, dec_d_in, emb_size, n_heads, n_hidden, ffn_n_hidden, num_layers, norm_first)
        self.lin_mu = nn.Linear(emb_size, d_out)
        self.lin_sigma = nn.Linear(emb_size, d_out)

    def forward(self, src_X: torch.Tensor, src_fX: torch.Tensor, tgt_X: torch.Tensor, tgt_fX: torch.Tensor, mask: torch.Tensor = None):
        Z = super().forward(src_X, src_fX, tgt_X, tgt_fX, mask)
        # dense layer to project to d_out dims
        # (batch_size, seq_len, emb_size) -> (batch_size, seq_len, d_out)
        return self.lin_mu(Z), F.softplus(self.lin_sigma(Z))