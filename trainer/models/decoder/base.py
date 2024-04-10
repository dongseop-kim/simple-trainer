import torch.nn as nn
from torch import Tensor as T


class BaseDecoder(nn.Module):
    def __init__(self, in_channels: int | list[int], in_strides: int | list[int]):
        super().__init__()
        self.in_channels = in_channels
        self.in_strides = in_strides
        self.out_channels: int | list[int] = -1
        self.out_strides: int | list[int] = -1

    def forward(self, x: T | list[T]) -> T | list[T]:
        raise NotImplementedError
