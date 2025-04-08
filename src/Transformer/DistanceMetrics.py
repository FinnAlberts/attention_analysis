import torch

def euclidean(q, k):
    """
    Compute the euclidean distance between all pairs of vectors q and k using Torch.

    Parameters:
    q (torch.Tensor): Tensor of shape (batch_size, seq_len, dim)
    k (torch.Tensor): Tensor of shape (batch_size, seq_len, dim)

    Returns:
    torch.Tensor: Pairwise distances of shape (batch_size, seq_len, seq_len)
    """
    return torch.cdist(q, k, p=2)

def manhattan(q, k):
    """
    Compute the manhattan distance between all pairs of vectors q and k using Torch.

    Parameters:
    q (torch.Tensor): Tensor of shape (batch_size, seq_len, dim)
    k (torch.Tensor): Tensor of shape (batch_size, seq_len, dim)

    Returns:
    torch.Tensor: Pairwise distances of shape (batch_size, seq_len, seq_len)
    """
    return torch.cdist(q, k, p=1)

def cosine(q, k):
    """
    Compute the cosine distance between all pairs of vectors vectors q and k using Torch.

    Parameters:
    q (torch.Tensor): Tensor of shape (batch_size, seq_len, dim)
    k (torch.Tensor): Tensor of shape (batch_size, seq_len, dim)

    Returns:
    torch.Tensor: Pairwise distances of shape (batch_size, seq_len, seq_len)
    """
    q_norm = torch.norm(q, p=2, dim=-1, keepdim=True)
    k_norm = torch.norm(k, p=2, dim=-1, keepdim=True)
    return 1 - torch.bmm(q / q_norm, (k / k_norm).transpose(1, 2))

def infinity_norm(q, k):
    """
    Compute the infinity norm distance between all pairs of vectors q and k using Torch.

    Parameters:
    q (torch.Tensor): Tensor of shape (batch_size, seq_len, dim)
    k (torch.Tensor): Tensor of shape (batch_size, seq_len, dim)

    Returns:
    torch.Tensor: Pairwise distances of shape (batch_size, seq_len, seq_len)
    """
    return torch.cdist(q, k, p=float('inf'))