
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from strainer.models.layers import ConvBNReLU

from .base import BasePredictionHead


class CompactSegmentationHead(BasePredictionHead):
    """
    Efficient segmentation head with compact architecture.

    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels
        input_scales: Input feature scale/stride
        init_bias_prob: Initial probability for bias initialization
    """

    def __init__(self, num_classes: int,
                 input_channels: int | list[int],
                 input_scales: int | list[int],
                 init_bias_prob: float = 0.01):
        # Ensure single input
        if isinstance(input_channels, list) and len(input_channels) > 1:
            raise ValueError("SegmentationHead expects single input feature")

        input_channels = input_channels[0] if isinstance(input_channels, list) else input_channels
        input_scales = input_scales[0] if isinstance(input_scales, list) else input_scales

        super().__init__(num_classes, input_channels, input_scales)

        # Create segmentation layers
        self.conv = nn.Sequential(ConvBNReLU(input_channels, input_channels, kernel_size=3),
                                  nn.Conv2d(input_channels, num_classes, kernel_size=1, bias=True),
                                  nn.Identity() if input_scales == 1 else
                                  nn.Upsample(scale_factor=input_scales, mode='bilinear'))

        # Initialize bias for focal loss if needed
        if init_bias_prob > 0:
            bias = -torch.log(torch.tensor((1.0 - init_bias_prob) / init_bias_prob))
            nn.init.constant_(self.conv[1].bias, bias)

    def forward(self, features: list[Tensor], targets=None) -> dict[str, Tensor]:
        assert len(features) == 1, "SegmentationHead expects single input feature"
        logits = self.conv(features[0])
        predictions = torch.sigmoid(logits)
        return {'logits': logits, 'predictions': predictions}


class MultiScaleCompactSegmentationHead(CompactSegmentationHead):
    """
    Multi-scale segmentation head that processes features at multiple scales.

    Args:
        num_classes: Number of output classes
        input_channels: List of input channels for each scale
        input_scales: List of input scales/strides
        fusion_weights: Optional weights for feature fusion
        init_bias_prob: Initial probability for bias initialization
    """

    def __init__(self, num_classes: int,
                 input_channels: list[int],
                 input_scales: list[int],
                 fusion_weights: list[float] | None = None,
                 init_bias_prob: float = 0.01):
        if not isinstance(input_channels, list) or not isinstance(input_scales, list):
            raise ValueError("MultiScaleSegmentationHead requires list of channels and scales")

        super().__init__(num_classes=num_classes,
                         input_channels=input_channels[0],  # Pass first channel for parent init
                         input_scales=1,  # Will handle scaling ourselves
                         init_bias_prob=init_bias_prob)

        # Multi-scale specific setup
        self.input_channels = input_channels
        self.input_scales = input_scales
        self.fusion_weights = (torch.tensor(fusion_weights) if fusion_weights
                               else torch.ones(len(input_channels)))

        # Create conv blocks for each scale
        self.convs = nn.ModuleList([nn.Sequential(ConvBNReLU(channels, channels, kernel_size=3),
                                                  nn.Conv2d(channels, num_classes, kernel_size=1, bias=True))
                                    for channels in self.input_channels])

        # Initialize biases
        if init_bias_prob > 0:
            bias = -torch.log(torch.tensor((1.0 - init_bias_prob) / init_bias_prob))
            for conv_block in self.convs:
                nn.init.constant_(conv_block[-1].bias, bias)

    def forward(self, features: list[Tensor], targets=None) -> dict:
        # Process each scale
        logits = [F.interpolate(conv(feature), scale_factor=scale, mode='bilinear', align_corners=False)
                  for feature, conv, scale in zip(features, self.convs, self.input_scales)]

        # Generate predictions
        predictions = [torch.sigmoid(logit) for logit in logits]

        # Compute weighted average
        final_prediction = self._compute_weighted_fusion(predictions)

        return (logits, final_prediction) if self.return_logits else final_prediction

    def _compute_weighted_fusion(self, predictions: list[Tensor]) -> Tensor:
        """Compute weighted average of predictions."""
        if len(predictions) != len(self.fusion_weights):
            raise ValueError("Number of predictions must match number of weights")

        weighted_preds = [w * p for w, p in zip(self.fusion_weights, predictions)]
        return torch.sum(torch.stack(weighted_preds), dim=0) / (self.fusion_weights.sum() + 1e-8)
