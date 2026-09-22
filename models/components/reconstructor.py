import torch
import torch.nn as nn

from transformer import Transformer
from .utils.position_encoding import add_position_encoding, sin_cos_position_encoding

class Reconstructor(nn.Module):

    def __init__(
            self, 
            num_patches,
            patch_size,  
            embedding_dim, 
            n_blocks, 
            n_heads, 
            ffn_ratio,
            init_std=0.02
        ):
        super(Reconstructor, self).__init__()

        self.num_patches = num_patches
        self.patch_size = patch_size

        self.embedding_dim = embedding_dim

        self.transformer = Transformer(
            n_blocks,
            embedding_dim,
            n_heads,
            ffn_ratio,
            init_std
        )

        self.query_tokens = nn.Parameter(
            torch.zeros((1, 1, embedding_dim))
        )

        self.norm = nn.LayerNorm(embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)

        self._init_weights(init_std)

    def forward(self, x, invisible_idx):
        C, N = self.num_patches

        x = self.linear1(x)
        batch_size, _, embedding_dim = x.shape

        invisible_idx = invisible_idx.to(x.device)
        N_invisible_idx = invisible_idx.shape[0]

        query_position_embedding = sin_cos_position_encoding(seq_len=C * N, embedding_dim=embedding_dim, device=x.device)
        query_position_embedding = torch.index_select(query_position_embedding, dim=0, index=invisible_idx)
        query_position_embedding = query_position_embedding.unsqueeze(0).repeat(batch_size, 1, 1)
        
        query_tokens = self.query_tokens.repeat((batch_size, N_invisible_idx, 1))
        query_tokens = query_tokens + query_position_embedding

        x = x.flatten(1, 2)
        x = add_position_encoding(x, device=x.device)
        x = torch.cat([query_tokens, x], dim=1)

        x = self.transformer(x)

        x = x[:, :N_invisible_idx, :]
        x = self.norm(x)
        x = self.linear2(x)

        return x

    def _init_weights(self, init_std):
        nn.init.trunc_normal_(self.query_tokens, std=init_std)

        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0)

        nn.init.trunc_normal_(self.linear1.weight, std=init_std)
        nn.init.constant_(self.linear1.bias, 0)
        
        nn.init.trunc_normal_(self.linear2.weight, std=init_std)
        nn.init.constant_(self.linear2.bias, 0)