import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

from .components.encoder import Encoder
from .components.predictor import Predictor
from .components.reconstructor import Reconstructor
from .components.utils.apply_mask import apply_visible_mask

class MaskAutoEncoder(nn.Module):

    def __init__(self, models_configs, use_part_pred=True, device='cuda'):
        super(MaskAutoEncoder, self).__init__()

        self.device = device
        self.use_part_pred = use_part_pred
        
        encoder = Encoder(**models_configs['encoder'])
        predictor = Predictor(use_part_pred=use_part_pred, **models_configs['predictor'])
        reconstructor = Reconstructor(**models_configs['reconstructor'])
        
        momentum_encoder = copy.deepcopy(self.encoder)
        for param in momentum_encoder.parameters():
            param.requires_grad = False

        self.encoder = encoder
        self.momentum_encoder = momentum_encoder
        self.predictor = predictor
        self.reconstructor = reconstructor

        self.num_patches = self.encoder.num_patches

        self.to(device)
    
    def forward(self, x, visible_idx, invisible_idx):
        z = self.encoder(x, visible_idx)
        z, combine_z = self.predictor(z, visible_idx)
        if not self.use_part_pred:
            combine_z = z
        r = self.reconstructor(combine_z, invisible_idx)
        
        return z, r

    def forward_momentum(self, x, invisible_idx):
        C, N = self.num_patches
        patch_size_c, patch_size_n = x.shape[-2] // C, x.shape[-1] // N
        
        with torch.no_gard():
            h = self.momentum_encoder(x)
            h = F.layer_norm(h, (h.size(-1), ))

            x = x.view(x.shape[0], C, patch_size_c, N, patch_size_n)
            x = x.permute(0, 3, 1, 2, 4).contiguous()
            x = x.view(x.shape[0], C, N, patch_size_c * patch_size_n)

            y = apply_visible_mask(x, invisible_idx.to(x.device))
            y = F.layer_norm(y, (y.size(-1), ))

            return h, y