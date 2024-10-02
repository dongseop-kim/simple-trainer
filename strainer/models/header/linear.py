from typing import Iterable, Optional

import torch
import torch.nn as nn
from torch import Tensor as T

from strainer.models.header.base import BaseHeader
from strainer.models.layers import ConvBNReLU


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


class ConciseSegmentor(BaseHeader):
    """
    Concise Segmentation Header that includes a 3x3 conv-bn-relu, a 1x1 conv and a bilinear upsampling.
    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of input channels.
        in_strides (int): Input strides.
        return_logit (bool): Whether to return the logit or not. Default is False.
        init_prob (float): Initial probability for the last conv layer. Default is 0.01. this is used for focal loss.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Output logits and the sigmoid output. If return_logit is False, only the sigmoid output is returned.
    """

    def __init__(self, num_classes: int, in_channels: int, in_strides: int, return_logits=False, init_prob: float = 0.01):

        # inchannels 가 list인데 길이가 1보다 크면 에러를 발생시킨다.
        if isinstance(in_channels, list) and len(in_channels) > 1:
            raise ValueError(f'LinearHeader only supports single input')

        in_channels = in_channels[0] if isinstance(in_channels, list) else in_channels
        super().__init__(num_classes=num_classes, in_channels=in_channels,
                         in_strides=in_strides, return_logits=return_logits)

        self.conv = nn.Sequential(ConvBNReLU(in_channels, in_channels, 3, 1, 1),
                                  nn.Conv2d(in_channels, num_classes, 1, 1, 0, bias=True),
                                  nn.Identity() if in_strides == 1 else nn.Upsample(scale_factor=in_strides, mode='bilinear'))

        # it is selected from focal loss bias setting
        self.initial_prob = T((1.0 - init_prob) / init_prob)
        nn.init.constant_(self.conv[-2].bias, -torch.log(self.initial_prob))

    def forward(self, x: T | list[T]) -> tuple[T, T] | T:
        logits = self.conv(x[0] if isinstance(x, list) else x)  # always single input
        output = torch.sigmoid(logits)
        return (logits, output) if self.return_logits else output
