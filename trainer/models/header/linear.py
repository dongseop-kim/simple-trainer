import torch.nn as nn
from torch.nn.functional import sigmoid
from torch import Tensor as T

from .base import BaseHeader


class LinearHeader(BaseHeader):
    def __init__(self, num_classes: int,
                 in_channels: int | list[int],
                 in_strides: int,
                 return_logits: bool = False):
        # inchannels 가 list인데 길이가 1보다 크면 에러를 발생시킨다.
        if isinstance(in_channels, list) and len(in_channels) > 1:
            raise ValueError(f'LinearHeader only supports single input')
        in_channels = in_channels[0] if isinstance(in_channels, list) else in_channels
        super().__init__(num_classes=num_classes, in_channels=in_channels, in_strides=in_strides)
        self.linear = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.return_logits = return_logits

    def forward(self, x: list[T], target=None) -> T | tuple[T, T]:
        logits: T = self.linear(x[0])  # always single input
        output: T = sigmoid(logits)
        return (logits, output) if self.return_logits else output
