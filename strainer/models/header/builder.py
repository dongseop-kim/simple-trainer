from typing import List

from .base import BaseHeader
from .identity import IdentityHeader
from .linear import SingleConvHeader
from .segmentation import (CompactSegmentationHead,
                           MultiScaleCompactSegmentationHead)

available_headers = {'identity': IdentityHeader,
                     'singleconv': SingleConvHeader,
                     'compact_segmentation': CompactSegmentationHead,
                     'multi_scale_compact_segmentation': MultiScaleCompactSegmentationHead,
                     }


def build_header(name: str, num_classes: int,
                 in_channels: int | List[int], in_strides: int | List[int],
                 **kwargs) -> BaseHeader:
    """
    Build a header module based on the given name and parameters.

    Args:
        name (str): The name of the header to build.
        num_classes (int): Number of classes for the header.
        in_channels (int | List[int]): Number of input channels.
        in_strides (int | List[int]): Input strides.
        **kwargs: Additional keyword arguments for the header.

    Returns:
        BaseHeader: An instance of the specified header.

    Raises:
        ValueError: If the specified header name is not found in available_headers.
    """
    if name not in available_headers:
        raise ValueError(f'Unknown header name: {name}. '
                         f'Available headers are: {", ".join(available_headers.keys())}')

    return available_headers[name](num_classes=num_classes,
                                   in_channels=in_channels,
                                   in_strides=in_strides,
                                   **kwargs)
