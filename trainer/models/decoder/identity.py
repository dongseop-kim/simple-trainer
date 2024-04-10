from torch import Tensor as T

from models.decoder.base import BaseDecoder


class IdentityDecoder(BaseDecoder):
    """
    Identity decoder that does not change the input.

    Args:
        in_channels (Union[int, List[int]]): number of channels for each feature map that is passed to the module
        in_strides (Union[int, List[int]]): stride of each feature map that is passed to the module
    """

    def __init__(self, in_channels: int | list[int], in_strides: int | list[int]):
        super().__init__(in_channels, in_strides)
        self.out_channels = in_channels
        self.out_strides = in_strides

    def forward(self, x: T | list[T]) -> T | list[T]:
        return x
