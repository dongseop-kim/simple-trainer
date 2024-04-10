from functools import partial
from typing import List

import torch
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation


class UpsampleAdd(nn.Module):
    """
    Upsample and add two tensors.

    Args:
        scale_factor (int): scale factor
        mode (str): upsampling mode
        align_corners (bool): align_corners

    Returns:
        torch.Tensor: upsampled and added tensor    
    """

    def __init__(self, scale_factor: int, mode: str = 'bilinear', align_corners: bool = True):
        super().__init__()
        self.scale_factor = scale_factor
        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode=mode, align_corners=align_corners)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.scale_factor == 1:
            return x + y
        return self.upsample(x) + y


class Concat(nn.Module):
    """
    Concatenate tensors.

    Args:
        dim (int): dimension to concatenate

    Returns:
        torch.Tensor: concatenated tensor    
    """

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, dim=self.dim)


# w/o normalization layer
ConvReLU = partial(Conv2dNormActivation, norm_layer=None, activation_layer=nn.ReLU)
ConvSwish = partial(Conv2dNormActivation, norm_layer=None, activation_layer=nn.SiLU)
# w/ normalization layer
ConvBN = partial(Conv2dNormActivation, norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity)
ConvBNReLU = partial(Conv2dNormActivation, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
ConvBNSwish = partial(Conv2dNormActivation, norm_layer=nn.BatchNorm2d, activation_layer=nn.SiLU)
