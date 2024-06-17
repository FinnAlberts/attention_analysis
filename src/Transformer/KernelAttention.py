import torch
from torch import nn
from torch.nn import functional as F

### helper functions
def magnitude_term(query: torch.Tensor, keys: torch.Tensor, d: int, p_norm: int):
    norm = torch.pow(torch.norm(input=query, dim=-1, keepdim=True, p=p_norm), 2) + \
           torch.pow(torch.norm(input=keys, dim=-1, keepdim=True, p=p_norm), 2).transpose(-2, -1)
    norm = norm / (2 * d ** 0.5)
    return norm

"""
Linear kernel attention
"""
class LinearKernelAttention(nn.Module):

    def __init__(self, dropout_rate: float = 0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.attention_weight = None

    def forward(self, query: torch.Tensor, keys: torch.Tensor, vals: torch.Tensor, mask: torch.Tensor = None):
        # query/keys: (batch_size, n_heads, seq_len, n_hidden/n_heads)
        # vals:       (batch_size, n_heads, seq_len, n_hidden/n_heads)
        _, _, seq_len, d = query.shape

        # (batch_size, n_heads, seq_len, seq_len)
        preattn = query @ keys.transpose(-2, -1) + 1e-5

        if mask is not None:
            preattn = preattn.masked_fill(mask == 0, 0.0)

        self.attention_weight = F.normalize(preattn, dim=-1)

        # out: (batch_size, n_heads, seq_len, n_hidden/n_heads)
        return self.dropout(self.attention_weight @ vals)


"""
Squared Exponential kernel attention
This is the standard dot product attention mechanism reformulated as a similarity term and a magnitude term.
The formulation allows us to put more focus on 
params:
    - p_norm_sim: controls L^p norm of the difference between query and keys
    - p_norm_mag: controls L^p norm of the individual magnitudes of query and keys vectors
"""
class SEKernelAttention(nn.Module):

    def __init__(self, dropout_rate: float = 0.1, p_norm_sim: int = 2, p_norm_mag: int = 2, include_magnitude: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.p_norm_sim = p_norm_sim
        self.p_norm_mag = p_norm_mag
        if include_magnitude:
            self.magnitude = lambda x, q, k, d: x + magnitude_term(q, k, d, self.p_norm_mag)
        else:
            self.magnitude = lambda x, q, k, d: x
        self.attention_weight = None

    def forward(self, query: torch.Tensor, keys: torch.Tensor, vals: torch.Tensor, mask: torch.Tensor = None):
        # query/keys: (batch_size, n_heads, seq_len, n_hidden/n_heads = d)
        # vals:       (batch_size, n_heads, seq_len, n_hidden/n_heads = d)
        _, _, seq_len, d = query.shape

        # query: (batch_size, n_heads, seq_len, d) --> (batch_size, n_heads, d, 1, seq_len)
        # keys:  (batch_size, n_heads, seq_len, d) --> (batch_size, n_heads, d, seq_len, 1)
        # diff:  (batch_size, n_heads, d, 1, seq_len) - (batch_size, n_heads, d, seq_len, 1) -> (batch_size, n_heads, d, seq_len, seq_len)
        diff = (query[..., None].permute(0, 1, 3, 4, 2) - keys[..., None].permute(0, 1, 3, 2, 4))

        # preattn: (batch_size, n_heads, d, seq_len, seq_len) --> (batch_size, n_heads, seq_len, seq_len)
        preattn = -torch.norm(diff, dim=2, p=self.p_norm_sim) / (2 * d**0.5)

        # if magnitude is included, it is added to the post sine wave
        preattn = self.magnitude(preattn, query, keys, d)

        if mask is not None:
            preattn = preattn.masked_fill(mask == 0, float('-inf'))

        self.attention_weight = F.softmax(preattn, dim=-1)

        # out: (batch_size, n_heads, seq_len, key_dim)
        return self.dropout(self.attention_weight @ vals)


"""
Periodic kernel attention
includes magnitude (L2-norm) term of query and keys
params:
    - period: distance between consequtive repetitions
    - p_norm: controls the L^p norm (magnitude) of query and keys
    - include_magnitude: whether to include the magnitude term
"""
class PeriodicKernelAttention(nn.Module):

    def __init__(self, dropout_rate: float = 0.1, period: float = 1, p_norm: int = 2, include_magnitude: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.period = period
        self.p_norm = p_norm
        if include_magnitude:
            self.magnitude = lambda x, q, k, d: x + magnitude_term(q, k, d, self.p_norm)
        else:
            self.magnitude = lambda x, q, k, d: x
        self.attention_weight = None

    def forward(self, query: torch.Tensor, keys: torch.Tensor, vals: torch.Tensor, mask: torch.Tensor = None):
        # query/keys: (batch_size, n_heads, seq_len, n_hidden/n_heads = d)
        # vals:       (batch_size, n_heads, seq_len, n_hidden/n_heads = d)
        _, _, seq_len, d = query.shape

        query_norm = query / torch.norm(query, dim=-1, keepdim=True)
        keys_norm = keys / torch.norm(keys, dim=-1, keepdim=True)

        presine = torch.pi * torch.sqrt(2 - 2 * (query_norm @ keys_norm.transpose(-2, -1))) / self.period
        postsine = -2 * torch.sin(presine)**2 / d**0.5

        # if magnitude is included, it is added to the post sine wave
        preattn = self.magnitude(postsine, query, keys, d)

        if mask is not None:
            preattn = preattn.masked_fill(mask == 0, float('-inf'))

        self.attention_weight = F.softmax(preattn, dim=-1)

        # out: (batch_size, n_heads, seq_len, d)
        return self.dropout(self.attention_weight @ vals)

"""
Locally periodic kernel attention
includes magnitude (L2-norm) term of query and keys
params:
    - period: distance between consequtive repetitions
    - p_norm: controls the L^p norm (magnitude) of query and keys
    - include_magnitude: whether to include the magnitude term
"""
class LocallyPeriodicKernelAttention(nn.Module):

    def __init__(self, dropout_rate: float = 0.1, period: float = 1, p_norm: int = 2, include_magnitude: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.period = period
        self.p_norm = p_norm
        if include_magnitude:
            self.magnitude = lambda x, q, k, d: x + magnitude_term(q, k, d, self.p_norm)
        else:
            self.magnitude = lambda x, q, k, d: x
        self.attention_weight = None

    def forward(self, query: torch.Tensor, keys: torch.Tensor, vals: torch.Tensor, mask: torch.Tensor = None):
        # query/keys: (batch_size, n_heads, seq_len, n_hidden/n_heads = d)
        # vals:       (batch_size, n_heads, seq_len, n_hidden/n_heads = d)
        _, _, seq_len, d = query.shape

        query_norm = query / torch.norm(query, dim=-1, keepdim=True)
        keys_norm = keys / torch.norm(keys, dim=-1, keepdim=True)

        presine = torch.pi * torch.sqrt(2 - 2 * (query_norm @ keys_norm.transpose(-2, -1))) / self.period
        postsine = -2 * torch.sin(presine)**2 / d**0.5

        # normalised squared exponential term
        SE_norm = query_norm @ keys_norm.transpose(-2, -1)

        # if magnitude is included, it is added to the post sine wave
        preattn = self.magnitude(postsine, query, keys, d) + SE_norm

        if mask is not None:
            preattn = preattn.masked_fill(mask == 0, float('-inf'))

        self.attention_weight = F.softmax(preattn, dim=-1)

        # out: (batch_size, n_heads, seq_len, d)
        return self.dropout(self.attention_weight @ vals)

"""
Rational quadratic kernel attention
params:
    - alpha: controls L^p norm of the difference between query and keys
    - p_norm: controls the L^p norm (magnitude) of query and keys
    - include_magnitude: whether to include the magnitude term
"""
class RationalQuadraticKernelAttention(nn.Module):

    def __init__(self, dropout_rate: float = 0.1, alpha: int = 1, p_norm: int = 2, include_magnitude: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.alpha = alpha
        self.p_norm = p_norm
        if include_magnitude:
            self.magnitude = lambda x, q, k, d: x + magnitude_term(q, k, d, self.p_norm)
        else:
            self.magnitude = lambda x, q, k, d: x
        self.attention_weight = None

    def forward(self, query: torch.Tensor, keys: torch.Tensor, vals: torch.Tensor, mask: torch.Tensor = None):
        # query/keys: (batch_size, n_heads, seq_len, n_hidden/n_heads = d)
        # vals:       (batch_size, n_heads, seq_len, n_hidden/n_heads = d)
        _, _, seq_len, d = query.shape

        query_norm = query / torch.norm(query, dim=-1, keepdim=True)
        keys_norm = keys / torch.norm(keys, dim=-1, keepdim=True)

        # rational quadratic term
        RQ_term = 1 + (1 - query_norm @ keys_norm.transpose(-2, -1)) / (self.alpha * d**0.5)
        RQ_term = RQ_term**-self.alpha

        # if magnitude is included, it is added to the log RQ_term
        preattn = self.magnitude(torch.log(RQ_term + 1e-5), query, keys, d)

        if mask is not None:
            preattn = preattn.masked_fill(mask == 0, float('-inf'))

        self.attention_weight = F.softmax(preattn, dim=-1)

        # out: (batch_size, n_heads, seq_len, key_dim)
        return self.dropout(self.attention_weight @ vals)

"""
Implicit Kernel attention
this is an approximation of a continuous stationary kernel with monte carlo integration 
and R sampled spectral points, $w_r$, and Fourier feature map, $\phi_r$.
params:
    - R_features: dimensionality of the Fourier feature map
    - p_norm: controls the L^p norm (magnitude) of query and keys
    - include_magnitude: whether to include the magnitude term
"""
class ImplicitKernelAttention(nn.Module):

    def __init__(self, dropout_rate: float = 0.1, R_features: int = 64, p_norm: int = 2, include_magnitude: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.R_features = R_features
        self.p_norm = p_norm
        self.W = nn.LazyLinear(R_features)
        if include_magnitude:
            self.magnitude = lambda x, q, k, d: x + magnitude_term(q, k, d, self.p_norm)
        else:
            self.magnitude = lambda x, q, k, d: x
        self.attention_weight = None

    def forward(self, query: torch.Tensor, keys: torch.Tensor, vals: torch.Tensor, mask: torch.Tensor = None):
        # query/keys: (batch_size, n_heads, seq_len, n_hidden/n_heads = d)
        # vals:       (batch_size, n_heads, seq_len, n_hidden/n_heads = d)
        _, _, seq_len, d = query.shape

        # W_query/keys: (batch_size, n_heads, seq_len, R_feat)
        W_query = self.W(query)
        W_keys = self.W(keys)

        # phi_query/keys: (batch_size, n_heads, seq_len, 2*R_feat)
        phi_query = torch.cat([torch.cos(W_query), torch.sin(W_query)], dim=-1)
        phi_keys = torch.cat([torch.cos(W_keys), torch.sin(W_keys)], dim=-1)

        # energy: (batch_size, n_heads, seq_len, seq_len)
        energy = phi_query @ phi_keys.transpose(-2, -1)
        energy = energy / self.R_features
        energy = torch.mul(energy, energy)

        # if magnitude is included, it is added to the log energy
        preattn = self.magnitude(torch.log(energy + 1e-5), query, keys, d)

        if mask is not None:
            preattn = preattn.masked_fill(mask == 0, float('-inf'))

        self.attention_weight = F.softmax(preattn, dim=-1)

        # out: (batch_size, n_heads, seq_len, key_dim)
        return self.dropout(self.attention_weight @ vals)


"""
Change-point Kernel attention
The SE and Periodic kernels are combined via sigmoid functions that act as gates. 
params:
    - period: distance between consequtive repetitions
    - p_norm_sim: controls L^p norm of the difference between query and keys
    - p_norm_mag: controls L^p norm of the individual magnitudes of query and keys vectors
"""
class ChangePointKernelAttention(nn.Module):

    def __init__(self, dropout_rate: float = 0.1, period: float = 1, p_norm_sim: int = 2, p_norm_mag: int = 2, include_magnitude: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.period = period
        self.p_norm_sim = p_norm_sim
        self.p_norm_mag = p_norm_mag
        if include_magnitude:
            self.magnitude = lambda x, q, k, d: x + magnitude_term(q, k, d, self.p_norm_mag)
        else:
            self.magnitude = lambda x, q, k, d: x
        self.attention_weight = None

    def forward(self, query: torch.Tensor, keys: torch.Tensor, vals: torch.Tensor, mask: torch.Tensor = None):
        # query/keys: (batch_size, n_heads, seq_len, n_hidden/n_heads = d)
        # vals:       (batch_size, n_heads, seq_len, n_hidden/n_heads = d)
        _, _, seq_len, d = query.shape

        # query: (batch_size, n_heads, seq_len, d) --> (batch_size, n_heads, d, 1, seq_len)
        # keys:  (batch_size, n_heads, seq_len, d) --> (batch_size, n_heads, d, seq_len, 1)
        # diff:  (batch_size, n_heads, d, 1, seq_len) - (batch_size, n_heads, d, seq_len, 1) -> (batch_size, n_heads, d, seq_len, seq_len)
        diff = (query[..., None].permute(0, 1, 3, 4, 2) - keys[..., None].permute(0, 1, 3, 2, 4))

        # preattn: (batch_size, n_heads, d, seq_len, seq_len) --> (batch_size, n_heads, seq_len, seq_len)
        SE_term = -torch.norm(diff, dim=2, p=self.p_norm_sim) / (2 * d ** 0.5)

        # periodic term
        query_norm = query / torch.norm(query, dim=-1, keepdim=True)
        keys_norm = keys / torch.norm(keys, dim=-1, keepdim=True)
        presine = torch.pi * torch.sqrt(2 - 2 * (query_norm @ keys_norm.transpose(-2, -1))) / self.period
        periodic_term = -2 * torch.sin(presine) ** 2 / d ** 0.5

        # sigmoid term (we use normalised query and key to avoid vanishing gradients)
        sigm_query = torch.sigmoid(query_norm)
        sigm_keys  = torch.sigmoid(keys_norm.transpose(-2, -1))

        preattn = sigm_query @ sigm_keys * torch.exp(SE_term) + \
                  (1-sigm_query) @ (1-sigm_keys) * torch.exp(periodic_term)

        # if magnitude is included, it is added to the log energy
        preattn = self.magnitude(preattn, query, keys, d)

        if mask is not None:
            preattn = preattn.masked_fill(mask == 0, 0.0)

        self.attention_weight = F.normalize(preattn, dim=-1)

        # out: (batch_size, n_heads, seq_len, key_dim)
        return self.dropout(self.attention_weight @ vals)