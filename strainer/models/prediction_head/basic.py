from typing import Iterable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .base import BasePredictionHead


class BasicSingleConvHead(BasePredictionHead):
    """
    Simple prediction head with a single convolution layer.
    Supports optional dropout and spatial operations.

    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels
        input_scales: Input feature scale/stride
        dropout_rate: Dropout probability (default: 0.0)
        spatial_operation: Spatial operation ('pool' or 'interpolate')
        init_bias_prob: Initial probability for bias initialization
    """

    def __init__(self,
                 num_classes: int,
                 input_channels: int | list[int],
                 input_scales: int | list[int],
                 dropout_rate: float = 0.0,
                 spatial_operation: str | None = None,
                 init_bias_prob: float = 0.01):
        # Ensure single input
        if isinstance(input_channels, list) and len(input_channels) > 1:
            raise ValueError("SimpleConvolutionHead expects single input feature")

        input_channels = input_channels[0] if isinstance(input_channels, list) else input_channels
        input_scales = input_scales[0] if isinstance(input_scales, list) else input_scales

        super().__init__(num_classes, input_channels, input_scales)

        # Convolutional head with optional dropout and spatial operation
        self.conv = nn.Sequential(nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                                  nn.Conv2d(input_channels, num_classes, kernel_size=1),
                                  self._create_spatial_operation(spatial_operation))

        # Initialize bias for focal loss if needed
        if init_bias_prob > 0:
            bias = -torch.log(torch.tensor((1.0 - init_bias_prob) / init_bias_prob))
            nn.init.constant_(self.conv.bias, bias)

    def _create_spatial_operation(self, operation: str | None) -> nn.Module:
        """Create spatial operation module based on specification."""
        match operation:
            case 'pool':
                return nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     nn.Flatten(start_dim=1))
            case 'interpolate':
                return nn.Upsample(scale_factor=self.input_scales,
                                   mode='bilinear', align_corners=True)
            case _:
                return nn.Identity()

    def forward(self, features: list[Tensor], targets=None) -> dict[torch.Tensor]:
        assert len(features) == 1, "SimpleConvolutionHead expects single input feature"
        # Process
        logits = self.conv(features[0])
        predictions = torch.sigmoid(logits)

        return {'logits': logits, 'predictions': predictions}
