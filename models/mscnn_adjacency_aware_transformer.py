import torch
import torch.nn as nn 

from .components.multi_scale_cnn import MultiScaleCNN
from .components.adjacency_aware_transformer import AdjacencyAwareTransformer
from .utils.position_encoding import add_position_encoding

class MSCAAT(nn.Module):

    def __init__(
        self,
        adjacency,
        in_shape=[3, 102, 200],
        embedding_dim=128,
        hidden_dim=256,
        n_heads=16,
        n_transformer_blocks=6,
        kernel_sizes=[3, 6, 9],
        strides=[1, 1, 1],
        paddings=[0, 0, 0],
        if_average_pooling=True,
        ap_kernel_sizes=3,
        n_conv1_features=256,
        n_conv2_features=64,
        n_res_blocks=0,
        n_res_features=256,
        negative_slope=0.01,
        device='cuda'
    ):
        super(MSCAAT, self).__init__()
        self.device = device

        self.mscnn_temp = MultiScaleCNN(
            in_shape=in_shape,
            out_dim=embedding_dim,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            if_average_pooling=if_average_pooling,
            ap_kernel_size=ap_kernel_sizes,
            n_conv1_features=n_conv1_features,
            n_conv2_features=n_conv2_features,
            n_res_blocks=n_res_blocks,
            n_res_features=n_res_features,
            negative_slope=negative_slope
        )

        self.transformer_temp = AdjacencyAwareTransformer(
            n_blocks=n_transformer_blocks,
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            hidden_dim=hidden_dim,
            adjacency=adjacency
        )

        self.cls_temp = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.to(self.device)

    def forward(self, x_temp):
        batch_size = x_temp.size(0)

        embedding_temp = self.mscnn_temp(x_temp)

        cls_tokens_temp = self.cls_temp.expand(batch_size, -1, -1)
        
        embedding_temp = torch.cat((cls_tokens_temp, embedding_temp), dim=1)

        embedding_temp = add_position_encoding(embedding_temp, device=self.device)

        embedding_temp = self.transformer_temp(embedding_temp)

        return embedding_temp[:, 0, :]