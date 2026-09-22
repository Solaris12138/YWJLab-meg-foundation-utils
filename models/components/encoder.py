import torch
import torch.nn as nn

from transformer import Transformer
from patch_embedding import PatchEmbedding
from .utils.apply_mask import apply_visible_mask

class Encoder(nn.Module):
    
    def __init__(self, input_size, patch_size, patch_stride,
                 embedding_dim, n_blocks, n_heads, ffn_ratio, init_std=0.02):
        super(Encoder, self).__init__()

        self.embedding_dim = embedding_dim

        self.n_heads = n_heads
        self.n_blocks = n_blocks

        self.ffn_ratio = ffn_ratio

        self.patch_embedding = PatchEmbedding(
            input_size,
            patch_size,
            patch_stride,
            embedding_dim
        )
        self.num_patches = self.patch_embedding.num_patches

        self.transformer = Transformer(
            n_blocks,
            embedding_dim,
            n_heads,
            ffn_ratio,
            init_std
        )

        self.cls_token = nn.Parameters(
            torch.zeros((1, 1, embedding_dim))
        )
        
        self.norm = nn.LayerNorm(embedding_dim)

        self._init_weights(init_std)
    
    def forward(self, x, visible_idx=None):
        x = self.patch_embedding(x) # batch_size, n_channels, n_times -> batch_size, N, C, embedding_dim

        if visible_idx is not None:
            visible_idx = visible_idx.to(x.device)
            x = apply_visible_mask(x, visible_idx)

        batch_size, N, _, _ = x.shape

        x = x.flatten(0, 1)

        cls_token = self.cls_token.repeat((x.shape[0], 1, 1))
        x = torch.cat([cls_token, x], dim=1)

        x = self.transformer(x)
        x = x[:, 0, :]
        x = self.norm(x)

        x = x.reshape((batch_size, N, self.embedding_dim))

        return x

    def _init_weights(self, init_std):
        nn.init.trunc_normal_(self.cls_token, std=init_std)

        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0)