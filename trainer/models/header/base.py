from typing import List, Union

import torch.nn as nn


class BaseHeader(nn.Module):
    def __init__(self, num_classes: int,
                 in_channels: Union[int, List[int]],
                 in_strides: Union[int, List[int]]):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.in_strides = in_strides
