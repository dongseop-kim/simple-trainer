from typing import Dict, List, Type

from strainer.models.decoder.base import BaseDecoder
from strainer.models.decoder.identity import IdentityDecoder
from strainer.models.decoder.multi_scale import (ConvolutionalFeatureFusion,
                                                 MultiScaleFeatureFusion)

available_decoders: Dict[str, Type[BaseDecoder]] = {'identity': IdentityDecoder,
                                                    'multi_scale_fusion': MultiScaleFeatureFusion,
                                                    'convolutional_multi_scale_fusion': ConvolutionalFeatureFusion,
                                                    }


def build_decoder(name: str, in_channels: int | List[int], in_strides: int | List[int],
                  **kwargs) -> BaseDecoder:
    """
    Build a decoder module based on the given name and parameters.

    Args:
        name (str): The name of the decoder to build. Must be one of the keys in `available_decoders`.
        in_channels (int | List[int]): Number of input channels or list of input channels.
        in_strides (int | List[int]): Input stride or list of input strides.
        **kwargs: Additional keyword arguments to pass to the decoder constructor.

    Returns:
        BaseDecoder: An instance of the specified decoder.

    Raises:
        ValueError: If the specified decoder name is not found in available_decoders.
    """
    if name not in available_decoders:
        raise ValueError(f'Unknown decoder name: {name}. '
                         f'Available decoders are: {", ".join(available_decoders.keys())}')

    return available_decoders[name](in_channels=in_channels, in_strides=in_strides, **kwargs)
