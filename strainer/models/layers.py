from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops.misc import Conv2dNormActivation


class UpsampleAddition(nn.Module):
    """
    Upsamples a feature map and adds it to another feature map.
    Commonly used in feature pyramid networks and U-Net architectures.

    Args:
        scale_factor: Multiplier for spatial dimensions
        mode: Upsampling interpolation mode. Default: 'bilinear'
        align_corners: Whether to align corners in interpolation. Default: True

    Examples:
        >>> upsampler = FeatureUpsampler(scale_factor=2)
        >>> low_res = torch.randn(1, 64, 32, 32)
        >>> high_res = torch.randn(1, 64, 64, 64)
        >>> output = upsampler(low_res, high_res)  # (1, 64, 64, 64)
    """

    def __init__(self, scale_factor: int, mode: str = 'bilinear', align_corners: bool = True):
        super().__init__()
        self.scale_factor = scale_factor
        self.upsampler = nn.Upsample(scale_factor=scale_factor, mode=mode,
                                     align_corners=align_corners)

    def forward(self, low_res: Tensor, high_res: Tensor) -> Tensor:
        if self.scale_factor == 1:
            return low_res + high_res
        return self.upsampler(low_res) + high_res


class Concatenation(nn.Module):
    """
    Concatenates multiple feature tensors along specified dimension.
    Useful for combining features from different scales or sources.

    Args:
        dim: Dimension along which to concatenate. Default: 1 (channel dimension)

    Examples:
        >>> fusion = FeatureFusion(dim=1)
        >>> features = [torch.randn(1, 64, 32, 32), torch.randn(1, 32, 32, 32)]
        >>> output = fusion(features)  # (1, 96, 32, 32)
    """

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, features: list[Tensor]) -> Tensor:
        return torch.cat(features, dim=self.dim)


# Pre-configured convolution layers with different normalization and activation combinations
# w/o batch normalization
ConvoReLU = partial(Conv2dNormActivation, norm_layer=None, activation_layer=nn.ReLU)
ConvSwish = partial(Conv2dNormActivation, norm_layer=None, activation_layer=nn.SiLU)
ConvSiLU = partial(Conv2dNormActivation, norm_layer=None, activation_layer=nn.SiLU)

# w/ normalization layer
ConvBN = partial(Conv2dNormActivation, norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity)
ConvBNReLU = partial(Conv2dNormActivation, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
ConvBNSwish = partial(Conv2dNormActivation, norm_layer=nn.BatchNorm2d, activation_layer=nn.SiLU)
ConvBNSiLU = partial(Conv2dNormActivation, norm_layer=nn.BatchNorm2d, activation_layer=nn.SiLU)
ConvDWBNReLU = partial(Conv2dNormActivation, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU,
                       groups=lambda in_channels: in_channels  # Depthwise convolution
                       )
