from typing import List

from trainer.models.decoder.identity import IdentityDecoder
from trainer.models.decoder.upsample_concat import UpsampleConcat

available_decoders = {'identity': IdentityDecoder,
                      'upsample_concat': UpsampleConcat,
                      }


def build_decoder(name: str, in_channels: int | List[int], in_strides: int | List[int], **kwargs):
    assert name in available_decoders, f'Unknown decoder name: {name}'
    return available_decoders[name](in_channels=in_channels, in_strides=in_strides, **kwargs)
