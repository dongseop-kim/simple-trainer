from typing import Type

from strainer.models.feature_decoder.base import BaseFeatureDecoder
from strainer.models.feature_decoder.identity import IdentityFeatureDecoder
from strainer.models.feature_decoder.multi_scale import MultiScaleFeatureDecoder, ConvolutionalMultiScaleDecoder

available_decoders: dict[str, BaseFeatureDecoder] = {'identity': IdentityFeatureDecoder,
                                                     'multi_scale_fusion': MultiScaleFeatureDecoder,
                                                     'convolutional_multi_scale_fusion': ConvolutionalMultiScaleDecoder,
                                                     }


def build_feature_decoder(name: str, input_channels: int | list[int], input_scales: int | list[int],
                          **kwargs) -> BaseFeatureDecoder:
    """
    Build a decoder module based on the given configuration.

    Args:
        name: Decoder architecture name from available_decoders
        input_channels: Number of channels for each input feature
        input_scales: Stride/scale of each input feature
        **kwargs: Additional configuration for specific decoder

    Returns:
        BaseFeatureDecoder: Configured decoder instance

    Raises:
        ValueError: If decoder name is not registered

    Examples:
        >>> # Single scale decoder
        >>> decoder = build_decoder('identity', input_channels=256, input_scales=32)
        >>> # Multi scale decoder
        >>> decoder = build_decoder('multi_scale', 
        ...                        input_channels=[256, 512, 1024],
        ...                        input_scales=[8, 16, 32])
    """
    if name not in available_decoders:
        raise ValueError(f"Unknown decoder: '{name}'. "
                         f"Available options: {', '.join(sorted(available_decoders.keys()))}")

    return available_decoders[name](input_channels=input_channels, input_scales=input_scales, **kwargs)
