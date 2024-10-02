from typing import Iterable, Optional

import torch
import torch.nn as nn
from torch import Tensor as T

from strainer.models.header.base import BaseHeader


class SingleConvHeader(BaseHeader):
    """
    A header module consisting of a single 1x1 convolution layer followed by optional pooling and interpolation.
    This header can be used for classification or segmentation tasks.

    Args:
        num_classes (int): Number of classes for the output.
        in_channels (int or list[int]): Number of input channels.
        in_strides (int): Input stride of the feature maps.
        dropout (float, optional): Dropout rate. Default is 0.0. apply before the last conv layer.
        pool (str, optional): Pooling type. Can be 'avg' for average pooling or 'max' for max pooling. Default is None.
        interpolate (bool, optional): Whether to interpolate the output to the input resolution. Default is False.
        return_logits (bool, optional): Whether to return the logits or the final output. Default is False.
        init_prob (float, optional): Initial probability for the last conv layer bias initialization. Default is 0.01.
    """

    def __init__(self, num_classes: int, in_channels: int | list[int], in_strides: int,
                 dropout: float = 0.0, pool: Optional[str] = None, interpolate: bool = False,
                 return_logits: bool = False, init_prob: float = 0.01):
        if isinstance(in_channels, Iterable) and len(in_channels) > 1:
            raise ValueError(f'SingleConvHeader only supports single input')
        in_channels = in_channels[0] if isinstance(in_channels, Iterable) else in_channels
        in_strides = in_strides[0] if isinstance(in_strides, Iterable) else in_strides
        super().__init__(num_classes=num_classes, in_channels=in_channels,
                         in_strides=in_strides, return_logits=return_logits)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        match pool:
            case 'avg':
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
            case 'max':
                self.pool = nn.AdaptiveMaxPool2d((1, 1))
            case None:
                self.pool = nn.Identity()
            case _:
                raise ValueError(f'Unknown pooling type: {pool}')

        self.interpolate = nn.Upsample(scale_factor=in_strides, mode='bilinear', align_corners=True) \
            if interpolate else nn.Identity()

        # it is selected from focal loss bias setting
        self.initial_prob = torch.tensor((1.0 - init_prob) / init_prob)
        nn.init.constant_(self.conv.bias, -torch.log(self.initial_prob))

    def forward(self, x: T | list[T], target=None) -> T | tuple[T, T]:
        x = x[0] if isinstance(x, (list, tuple)) else x
        x = x.view(x.size(0), x.size(1), 1, 1) if len(x.shape) == 2 else x
        x = self.dropout(x)
        logits = self.interpolate(self.pool(self.conv(x)))
        output: T = torch.sigmoid(logits)
        return (logits, output) if self.return_logits else output
