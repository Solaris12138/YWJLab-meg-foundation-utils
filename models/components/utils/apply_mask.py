import torch

def apply_visible_mask(x, visible_idx):
    """Apply visible mask to data.

    Parameters
    ----------
    x : torch.Tensor, shape (batch_size, N, C, embedding_dim)
        The data to be masked.
    visible_idx : torch.Tensor, shape (N_visible, [C_visible])
        The indices of patches which could be visible to the encoder model.
    
    Returns
    -------
    visible_x : numpy array, shape (batch_size, N_visible, C_visible, embedding_dim)

    """

    batch_size, N, C, embedding_dim = x.shape

    if len(visible_idx.shape) == 2:
        N_visible, C_visible = visible_idx.shape

        visible_idx_ = visible_idx.reshape(
            (1, N_visible * C_visible, 1)
        ).repeat(batch_size, 1, embedding_dim)

        visible_x = torch.gather(
            x.reshape((batch_size, N * C, embedding_dim)),
            dim=1,
            index=visible_idx_
        )
        visible_x = visible_x.contiguous().view((batch_size, N_visible, C_visible, embedding_dim))

    else:
        N_visible = visible_idx.shape[0]

        visible_idx_ = visible_idx.reshape(
            (1, N_visible, 1)
        ).repeat(batch_size, 1, embedding_dim)
        
        visible_x = torch.gather(
            x.reshape((batch_size, N * C, embedding_dim)),
            dim=1,
            index=visible_idx_
        )
    
    return visible_x