from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class BaseFeatureDecoder(ABC, nn.Module):
    """
    Base class for all feature decoder implementations.

    Args:
        input_channels: Number of channels for each input feature
        input_scales: Stride/scale of each input feature
    """

    def __init__(self, input_channels: int | list[int], input_scales: int | list[int]):
        super().__init__()
        self.input_channels = ([input_channels] if isinstance(input_channels, int) else input_channels)
        self.input_scales = ([input_scales] if isinstance(input_scales, int) else input_scales)

        # Will be set by child classes
        self.output_channels: int
        self.output_scales: int

    @abstractmethod
    def forward(self, features: Tensor | list[Tensor]) -> list[Tensor]:
        """Process input features through the decoder."""
        pass

    def freeze(self, freeze_bn: bool = True) -> None:
        """
        Freeze decoder parameters to prevent updates during training.

        Args:
            freeze_bn: Whether to also freeze batch normalization statistics.
                     If False, BN stats will continue updating during training.
        """
        self._is_frozen = True
        for param in self.parameters():
            param.requires_grad = False

        if freeze_bn:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()

    def unfreeze(self) -> None:
        """
        Unfreeze decoder parameters to enable updates during training.
        Also sets batch normalization layers back to training mode.
        """
        self._is_frozen = False
        for param in self.parameters():
            param.requires_grad = True

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.train()

    def is_frozen(self) -> bool:
        """
        Check if decoder is currently frozen.

        Returns:
            bool: True if decoder is frozen, False otherwise
        """
        return self._is_frozen
