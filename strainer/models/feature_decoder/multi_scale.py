import torch
import torch.nn as nn
from torch import Tensor

from strainer.models.feature_decoder.base import BaseFeatureDecoder
from strainer.models.layers import ConvBNReLU


class MultiScaleFeatureDecoder(BaseFeatureDecoder):
    """
    Decoder that fuses multi-scale features through upsampling and concatenation.
    Commonly used in FPN-like architectures.

    Args:
        input_channels: List of channel counts for each input feature
        input_scales: List of scales/strides for each feature
        output_scales: Target output scale. Defaults to smallest input scale

    Examples:
        >>> decoder = MultiScaleFeatureDecoder(
        ...     input_channels=[256, 512, 1024],
        ...     input_scales=[8, 16, 32]
        ... )
        >>> features = [
        ...     torch.randn(1, 256, 64, 64),
        ...     torch.randn(1, 512, 32, 32),
        ...     torch.randn(1, 1024, 16, 16)
        ... ]
        >>> out = decoder(features)  # (1, 1792, 64, 64)
    """

    def __init__(self, input_channels: list[int], input_scales: list[int],
                 output_scales: int | None = None, **kwargs):
        if len(input_channels) != len(input_scales):
            raise ValueError(f"Channel/scale length mismatch: {len(input_channels)} vs {len(input_scales)}")
        super().__init__(input_channels, input_scales)

        self.output_channels = sum(self.input_channels)  # Concatenated channels
        self.output_scales = output_scales or min(self.input_scales)  # Smallest input scale

        # Create upsamplers for each scale
        self.upsamplers = nn.ModuleDict({f'scale_{i}': (nn.Identity() if int(s/self.output_scales) == 1
                                                        else nn.Upsample(scale_factor=s/self.output_scales,
                                                                         mode='bilinear', align_corners=True))
                                         for i, s in enumerate(self.input_scales)})

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        if len(features) != len(self.input_channels):
            raise ValueError(f"Expected {len(self.input_channels)} features, got {len(features)}")
        scaled_features = [self.upsamplers[f"scale_{i}"](feature) for i, feature in enumerate(features)]
        return [torch.cat(scaled_features, dim=1)]


class ConvolutionalMultiScaleDecoder(MultiScaleFeatureDecoder):
    """
    Enhanced multi-scale decoder with additional convolution for channel reduction.

    Args:
        input_channels: List of channel counts for each input feature
        input_scales: List of scales/strides for each feature
        output_scales: Target output scale
        output_channels: Number of output channels after fusion

    Examples:
        >>> decoder = ConvolutionalMultiScaleDecoder(
        ...     input_channels=[256, 512, 1024],
        ...     input_scales=[8, 16, 32],
        ...     output_channels=256
        ... )
        >>> features = [
        ...     torch.randn(1, 256, 64, 64),
        ...     torch.randn(1, 512, 32, 32),
        ...     torch.randn(1, 1024, 16, 16)
        ... ]
        >>> out = decoder(features)  # (1, 256, 64, 64)
    """

    def __init__(self, input_channels: list[int], input_scales: list[int],
                 output_scales: int | None = None, output_channels: int = 256, **kwargs):
        super().__init__(input_channels, input_scales, output_scales)
        self.output_channels = output_channels

        self.fusion_conv = ConvBNReLU(sum(self.input_channels), self.output_channels,
                                      kernel_size=1, stride=1, padding=0)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        fused_features = super().forward(features)[0]
        return [self.fusion_conv(fused_features)]
