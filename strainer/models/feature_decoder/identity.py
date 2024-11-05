from torch import Tensor

from strainer.models.feature_decoder import BaseFeatureDecoder


class IdentityFeatureDecoder(BaseFeatureDecoder):
    """
    Simple decoder that passes through input features without modification.
    Useful as a baseline or when no feature processing is needed.

    Args:
        input_channels: Number of input channels
        input_scales: Input feature scales/strides

    Examples:
        >>> decoder = IdentityFeatureDecoder(256, 32)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> out = decoder(x)  # Same shape as input
    """

    def __init__(self, input_channels: int | list[int], input_scales: int | list[int], **kwargs):
        super().__init__(input_channels, input_scales)
        self.output_channels = self.input_channels
        self.output_scales = self.input_scales

    def forward(self, features: Tensor | list[Tensor]) -> list[Tensor]:
        return features if isinstance(features, list) else [features]
