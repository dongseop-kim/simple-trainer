from typing import List, Optional
import torch
import torch.nn as nn

from models.base_module import ConvBNReLU
from models.decoder.base import BaseDecoder


class MultiScaleFeatureFusion(BaseDecoder):
    """
    Fuse multiple features from different scales by upsampling and concatenation.

    Args:
        in_channels (List[int]): Number of input channels for each feature.
        in_strides (List[int]): List of input strides for each feature.
        out_strides (Optional[int]): Output stride. If None, it will be set to the first input stride.

    Returns:
        torch.Tensor: Output tensor with shape (N, out_channels, H, W).
    """

    def __init__(self, in_channels: List[int], in_strides: List[int],
                 out_strides: Optional[int] = None):
        if len(in_channels) != len(in_strides):
            raise ValueError("in_channels and in_strides should have the same length.")
        super().__init__(in_channels, in_strides)

        self.out_channels = sum(self.in_channels)
        self.out_strides = out_strides or self.in_strides[0]

        self.upsampler = nn.ModuleDict({f"decoder_{i}": (nn.Identity() if int(s / self.out_strides) == 1
                                                         else nn.Upsample(scale_factor=(s / self.out_strides),
                                                                          mode="bilinear",
                                                                          align_corners=True))
                                        for i, s in enumerate(self.in_strides)})

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        outs = [self.upsampler[f"decoder_{i}"](feature)
                for i, feature in enumerate(features)]
        return torch.cat(outs, dim=1)


class ConvolutionalFeatureFusion(MultiScaleFeatureFusion):
    """
    Fuse multiple features from different scales and apply convolution to reduce channels.

    Args:
        in_channels (List[int]): Number of input channels for each feature.
        in_strides (List[int]): List of input strides for each feature.
        out_channels (int): Number of output channels. Default is 256.
        out_strides (Optional[int]): Output stride. If None, it will be set to the first input stride.

    Returns:
        torch.Tensor: Output tensor with shape (N, out_channels, H, W).
    """

    def __init__(self,
                 in_channels: List[int],
                 in_strides: List[int],
                 out_strides: Optional[int] = None,
                 out_channels: int = 256):
        super().__init__(in_channels, in_strides, out_strides)
        self.out_channels = out_channels
        self.conv = ConvBNReLU(sum(self.in_channels), self.out_channels,
                               kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        out = super().forward(features)
        return self.conv(out)
