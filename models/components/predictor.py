import torch
import torch.nn as nn

from transformer import Transformer
from .utils.position_encoding import add_position_encoding

class Predictor(nn.Module):

    def __init__(
            self, 
            num_patches,  
            embedding_dim, 
            n_blocks, 
            n_heads, 
            ffn_ratio,
            use_part_pred=True,
            init_std=0.02
        ):
        super(Predictor, self).__init__()

        self.num_patches = num_patches

        self.embedding_dim = embedding_dim

        self.n_heads = n_heads
        self.n_blocks = n_blocks

        self.ffn_dim = ffn_ratio

        self.use_part_pred = use_part_pred

        self.transformer = Transformer(
            n_blocks,
            embedding_dim,
            n_heads,
            ffn_ratio,
            init_std
        )

        self.mask_tokens = nn.Parameter(
            torch.zeros((1, 1, embedding_dim))
        )

        self.norm = nn.LayerNorm(embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)

        self._init_weights(init_std)

    def forward(self, x, visible_idx):
        C, N = self.num_patches

        if self.use_part_pred:
            input_x = x
        
        x = self.linear1(x)

        N_visible_idx = torch.floor(visible_idx[:, 0] / C).long().to(x.device)
        N_invisible_idx = torch.tensor(
            list(
                set(list(range(0, N))) - set(N_visible_idx.tolist())
            )
        ).to(x.device)

        mask_tokens = self.mask_tokens.repeat((x.shape[0], N_invisible_idx.shape[0], 1))
        x = torch.cat([x, mask_tokens], dim=1)

        mask_idx = torch.concat([N_visible_idx, N_invisible_idx], dim=0)
        x = torch.index_select(x, dim=1, index=torch.argsort(mask_idx))

        x = add_position_encoding(x, device=x.device)

        x = self.transformer(x)
        x = self.norm(x)
        x = self.linear2(x)

        combine_x = None
        if self.use_part_pred:
            combine_x = torch.index_select(x, dim=1, index=N_invisible_idx)
            combine_x = torch.concat([input_x, combine_x], dim=1)
            combine_x = torch.index_select(combine_x, dim=1, index=torch.argsort(mask_idx))

        return x, combine_x

    def _init_weights(self, init_std):
        nn.init.trunc_normal_(self.mask_tokens, std=init_std)

        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0)

        nn.init.trunc_normal_(self.linear1.weight, std=init_std)
        nn.init.constant_(self.linear1.bias, 0)
        
        nn.init.trunc_normal_(self.linear2.weight, std=init_std)
        nn.init.constant_(self.linear2.bias, 0)