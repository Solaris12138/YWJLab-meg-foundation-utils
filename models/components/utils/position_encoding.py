import torch
import math

def sin_cos_position_encoding(seq_len, embedding_dim, device='cuda'):
    """Compute sin-cos position embedding for a given sequence input.

    Parameters
    ----------
    seq_len : int
        The length of input sequence.
    embedding_dim : int
        The dimension of input embeddings.
    
    Returns
    -------
    pe : torch.Tensor, shape (seq_len, embedding_dim)

    """
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1).to(device)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embedding_dim)).to(device)

    pe = torch.zeros(seq_len, embedding_dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

def add_position_encoding(x, device='cuda'):
    """Add sin-cos position embedding to a given sequence input.

    Parameters
    ----------
    x : torch.Tensor, shape (batch_size, seq_len, embedding_dim)
        The input data.
    
    Returns
    -------
    x : torch.Tensor, shape (batch_size, seq_len, embedding_dim)

    """
    batch_size, seq_len, embedding_dim = x.size()

    position_encoding = sin_cos_position_encoding(
        seq_len,
        embedding_dim,
        device=device
    )

    position_encoding = position_encoding.unsqueeze(0).repeat(batch_size, 1, 1)

    return x + position_encoding