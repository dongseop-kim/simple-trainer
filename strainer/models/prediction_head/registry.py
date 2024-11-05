from typing import Type

from .base import BasePredictionHead
from .basic import BasicSingleConvHead
from .identity import IdentityHead
from .segmentation import (CompactSegmentationHead,
                           MultiScaleCompactSegmentationHead)

available_heads: dict[str, Type[BasePredictionHead]] = {'identity': IdentityHead,
                                                        'basic_single_conv': BasicSingleConvHead,
                                                        'compact_segmentation': CompactSegmentationHead,
                                                        'multi_scale_compact_segmentation': MultiScaleCompactSegmentationHead,
                                                        }


def build_prediction_head(name: str, num_classes: int,
                          input_channels: int | list[int],
                          input_scales: int | list[int],
                          **kwargs) -> BasePredictionHead:
    """
    Construct a prediction head based on configuration.

    Args:
        name: Head architecture name
        num_classes: Number of output classes
        input_channels: Number of channels in input features
        input_scales: Scale/stride of input features
        **kwargs: Additional head-specific configuration

    Returns:
        PredictionHead: Configured prediction head instance

    Raises:
        ValueError: If head architecture is not registered
    """
    if name not in available_heads:
        raise ValueError(f"Unknown head architecture: '{name}'. "
                         f"Available options: {', '.join(sorted(available_heads.keys()))}")

    return available_heads[name](num_classes=num_classes, input_channels=input_channels,
                                 input_scales=input_scales, **kwargs)
