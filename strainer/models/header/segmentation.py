
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor as T
from torch.nn import functional as F

from strainer.models.header.base import BaseHeader
from strainer.models.layers import ConvBNReLU


class CompactSegmentationHead(BaseHeader):
    """
    A compact and efficient segmentation head for neural networks.

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


class MultiScaleCompactSegmentationHead(CompactSegmentationHead):
    """
    A compact segmentation head that processes multi-scale features.

    This module takes a list of features as input and returns a list of segmentation maps
    that has the same length as the input list, along with their weighted average.

    Args:
        num_classes (int): Number of classes.
        in_channels (List[int]): Number of input channels for each scale.
        in_strides (List[int]): Input strides for each scale.
        weights (Optional[List[float]]): Weights for each input feature. Default is None.
        init_prob (float): Initial probability for the last conv layer. Default is 0.01. Used for focal loss.
        return_logits (bool): Whether to return logits along with the final output. Default is False.

    Returns:
        tuple[List[torch.Tensor], torch.Tensor] | torch.Tensor: 
            If return_logits is True, returns a tuple of output segmentation logit maps and their weighted average.
            Otherwise, returns only the weighted average of the output maps.
    """

    def __init__(self, num_classes: int,
                 in_channels: list[int],
                 in_strides: list[int],
                 weights: Optional[list[float]] = None,
                 init_prob: float = 0.01,
                 return_logits: bool = False):
        super().__init__(num_classes=num_classes, in_channels=in_channels[0], in_strides=1,
                         return_logits=return_logits, init_prob=init_prob)

        if not (isinstance(in_channels, list) and isinstance(in_strides, list)):
            raise ValueError("in_channels and in_strides must be lists.")

        self.in_channels = in_channels
        self.in_strides = in_strides
        self.weights = torch.Tensor(weights) if weights is not None else torch.ones(len(in_channels))

        self.multi_scale_convs = nn.ModuleList([nn.Sequential(ConvBNReLU(in_channel, in_channel, 3, 1, 1),
                                                              nn.Conv2d(in_channel, num_classes, 1, 1, 0, bias=True))
                                                for in_channel in self.in_channels])

        for conv in self.multi_scale_convs:
            nn.init.constant_(conv[-1].bias, -torch.log(self.initial_prob))

    def forward(self, features: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor] | torch.Tensor:
        logits = [conv(feature) for feature, conv in zip(features, self.multi_scale_convs)]
        logits = [F.interpolate(out, scale_factor=stride, mode='bilinear', align_corners=False)
                  for out, stride in zip(logits, self.in_strides)]

        outputs = [torch.sigmoid(logit) for logit in logits]
        weighted_output = self.get_weighted_average(outputs, self.weights)

        return (logits, weighted_output) if self.return_logits else weighted_output

    @staticmethod
    def get_weighted_average(pred: list[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        """
        Get weighted average of the input predictions.

        Args:
            pred (List[torch.Tensor]): List of predictions.
            weights (torch.Tensor): Tensor of weights.

        Returns:
            torch.Tensor: Weighted average of the input predictions.
        """
        if len(pred) != len(weights):
            raise ValueError("The length of pred and weights should be the same.")

        weighted_pred = [torch.mul(weight, p) for weight, p in zip(weights, pred)]
        weight_sum = torch.sum(weights) + 1e-8
        average_pred = torch.sum(torch.stack(weighted_pred, dim=0), dim=0) / weight_sum
        return average_pred
