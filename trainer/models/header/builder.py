from typing import List

from .identity import IdentityHeader
from .linear import LinearHeader, ConciseSegmentor


available_headers = {'identity': IdentityHeader,
                     'linear': LinearHeader,
                     'concise': ConciseSegmentor,
                     }


def build_header(name: str, num_classes: int, in_channels: int | List[int],
                 in_strides: int | List[int], **kwargs):
    assert name in available_headers, f'Unknown header name: {name}'
    return available_headers[name](num_classes=num_classes, in_channels=in_channels,
                                   in_strides=in_strides, **kwargs)
