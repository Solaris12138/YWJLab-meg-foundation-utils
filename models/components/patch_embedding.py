import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):

    def __init__(
        self,
        input_size,
        patch_size,
        patch_stride=None,
        embedding_dim=256
    ):
        super(PatchEmbedding, self).__init__()

        assert input_size[1] % patch_size == 0, \
            f'The shape of input data must be divisible by patch_size ({patch_size}) for the 2nd dimension, but got {input_size[1]}.'

        self.input_size = input_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.embedding_dim = embedding_dim

        if patch_stride is None:
            self.num_patches = (input_size[0], input_size[1] // patch_size)
        else:
            self.num_patches = (input_size[0], (input_size[1] - patch_size) // patch_stride + 1)

        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=embedding_dim,
            kernel_size=(1, patch_size),
            stride=(1, patch_size if patch_stride is None else patch_stride)
        )

    def forward(self, x):        
        x = x.unsqueeze(1)
        x = self.proj(x).transpose(1, 3)

        return x