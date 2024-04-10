from typing import Optional

import torch.nn as nn
from torch import Tensor as T

from trainer.models.decoder.base import BaseDecoder
from trainer.models.layers import Concat


class UpsampleConcat(BaseDecoder):
    """
    Upsample and concatenate all multiple features. 
    if out_strides is not specified, it will be set to the first feature's stride.
    else it will be set to the specified value.

    Args:
        in_channels (List[int]): Number of input channels.
        in_strides (List[int]): List of input strides.
        out_strides (Optional[int]): Output stride. if None, it will be set to the first input stride.

    Returns:
        torch.Tensor: Output tensor that has the shape of (N, out_channels, H, W).
    """

    def __init__(self,
                 in_channels: list[int],
                 in_strides: list[int],
                 out_strides: Optional[int] = None):
        if len(in_channels) != len(in_strides):
            raise ValueError("in_channels and in_strides should have the same length.")
        super().__init__(in_channels, in_strides)

        self.out_channels = sum(self.in_channels)
        self.out_strides = self.in_strides[0] if out_strides is None else out_strides

        self.upsampler = nn.ModuleDict({f"decoder_{i}": nn.Identity() if int(s / self.out_strides) == 1
                                        else nn.Upsample(scale_factor=(s / self.out_strides), mode="bilinear", align_corners=True)
                                        for i, s in enumerate(self.in_strides)})
        self.concat = Concat(dim=1)

    def forward(self, features: list[T]) -> T:
        outs: list[T] = []
        for i, feature in enumerate(features):
            out = self.upsampler[f"decoder_{i}"](feature)
            outs.append(out)
        return self.concat(outs)
