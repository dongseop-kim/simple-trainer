from .base import BasePredictionHead


class IdentityHead(BasePredictionHead):
    """
    Identity decoder that does not change the input.

    Args:
        in_channels (Union[int, List[int]]): number of channels for each feature map that is passed to the module
        in_strides (Union[int, List[int]]): stride of each feature map that is passed to the module
    """

    def __init__(self, num_classes: int, input_channels: int | list[int], input_scales: int | list[int]):
        super().__init__(num_classes=num_classes, input_channels=input_channels, input_scales=input_scales)

    def forward(self, x):
        return x
