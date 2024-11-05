from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class BasePredictionHead(ABC, nn.Module):
    """
    Base class for all prediction head implementations.

    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels for each feature level
        input_scales: Scale/stride of each input feature level
    """

    def __init__(self,
                 num_classes: int,
                 input_channels: int | list[int],
                 input_scales: int | list[int]):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_scales = input_scales

    @abstractmethod
    def forward(self, features: Tensor | list[Tensor], targets=None) -> dict[str, Tensor]:
        """
        Process features through the prediction head.
        All prediction heads include keys 'predictions' and 'logits' in the output dictionary.
        """
        pass
