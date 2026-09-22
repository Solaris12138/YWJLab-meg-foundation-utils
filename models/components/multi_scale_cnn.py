import torch
import torch.nn as nn

from functools import reduce
from operator import add

class ResBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        negative_slope=0.01
    ):
        super(ResBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=(1, 3),
                stride=1,
                padding=(0, 1)
            ),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=(1, 3),
                stride=1,
                padding=(0, 1)
            )
        )

    def forward(self, x):
        residual = x
        out = self.layer(x)
        out += residual
        return out

class MultiScaleCNN(nn.Module):

    def __init__(
        self,
        in_shape=[3, 102, 200],  #(3, n_sensor, n_timepoints/n_freqcomponents)
        out_dim=512,
        kernel_sizes=[3, 6, 9],
        strides=[1, 1, 1],
        paddings=[0, 0, 0],
        if_average_pooling=True,
        ap_kernel_size=3,
        n_conv1_features=256,
        n_conv2_features=64,
        n_res_blocks=0,
        n_res_features=256,
        negative_slope=0.01
    ):
        super(MultiScaleCNN, self).__init__()

        self.in_channels, self.in_seqlen = in_shape[0], in_shape[-1]
        self.out_dim = out_dim

        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings

        self.if_average_pooling = if_average_pooling
        self.ap_kernel_size = ap_kernel_size
        
        self.n_conv1_features = n_conv1_features
        self.n_conv2_features = n_conv2_features
        self.n_res_features = n_res_features
        self.n_res_blocks = n_res_blocks
        self.negative_slope = negative_slope

        self.layer_norm1, conv1_shape = self._init_layer_norm1()
        self.layer_norm2, conv2_shape = self._init_layer_norm2(conv1_shape)
        
        self.conv1 = self._init_conv1()
        self.conv2 = self._init_conv2()

        self.fc = nn.Linear(in_features=conv2_shape * n_conv2_features, out_features=out_dim, bias=True)

    def _init_conv1(self):
        conv1 = []
        for kernel_size, stride, padding in zip(self.kernel_sizes, self.strides, self.paddings):
            layers = [nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.n_conv1_features,
                kernel_size=(1, kernel_size),
                stride=(1, stride),
                padding=(0, padding)
            )]
            for _ in range(self.n_res_blocks):
                layers.append(
                    ResBlock(
                        in_channels=self.n_conv1_features, 
                        mid_channels=self.n_res_features, 
                        out_channels=self.n_conv1_features, 
                        negative_slope=self.negative_slope
                    )
                )
            layers.append(nn.LeakyReLU(negative_slope=self.negative_slope))
            if self.if_average_pooling:
                layers.append(
                    nn.AvgPool2d(
                        kernel_size=(1, self.ap_kernel_size),
                        stride=None,
                        padding=0,
                    )
                )
            conv1.append(nn.Sequential(*layers))
        return nn.ModuleList(conv1)
    
    def _init_conv2(self):
        conv2 = [
            nn.Conv2d(
                in_channels=self.n_conv1_features,
                out_channels=self.n_conv2_features,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=0
            ),
            nn.LeakyReLU(negative_slope=self.negative_slope),
        ]
        if self.if_average_pooling:
            conv2.append(
                nn.AvgPool2d(
                    kernel_size=(1, self.ap_kernel_size),
                    stride=None,
                    padding=0,
                )
            )
        return nn.Sequential(*conv2)
    
    def _init_layer_norm1(self):
        if self.if_average_pooling:
            conv1_shapes = [
                self.compute_size(
                    self.compute_size(
                        self.in_seqlen, 
                        kernel_size, 
                        stride, 
                        padding
                    ),
                    self.ap_kernel_size,
                    self.ap_kernel_size,
                    0
                ) for kernel_size, stride, padding in zip(self.kernel_sizes, self.strides, self.paddings)
            ]
        else:
            conv1_shapes = [
                self.compute_size(
                    self.in_seqlen, 
                    kernel_size, 
                    stride, 
                    padding
                ) for kernel_size, stride, padding in zip(self.kernel_sizes, self.strides, self.paddings)
            ]
        conv1_cat_shape = reduce(add, conv1_shapes)
        return nn.LayerNorm(normalized_shape=conv1_cat_shape, eps=1e-05, elementwise_affine=True, bias=True), conv1_cat_shape

    def _init_layer_norm2(self, in_shape):
        conv2_shape = self.compute_size(
            in_shape, 
            self.ap_kernel_size, 
            self.ap_kernel_size, 
            0
        ) if self.if_average_pooling else in_shape
        return nn.LayerNorm(normalized_shape=conv2_shape, eps=1e-05, elementwise_affine=True, bias=True), conv2_shape

    def forward(self, x):
        conv1_out = self.layer_norm1(
            torch.cat(tuple(conv(x) for conv in self.conv1), dim=-1)
        )

        conv2_out = self.layer_norm2(
            self.conv2(conv1_out)
        )
        
        conv2_out_ = torch.permute(conv2_out, dims=(0, 2, 1, 3))
        return self.fc(torch.flatten(conv2_out_, start_dim=-2, end_dim=-1))

    @staticmethod
    def compute_size(input_size, kernel_size, stride, padding):
        return (input_size + 2*padding - kernel_size) // stride + 1