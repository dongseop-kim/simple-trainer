from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor as T

from .decoder import BaseDecoder, build_decoder
from .encoder import build_encoder
from .header import BaseHeader, build_header


class Model(nn.Module):
    def __init__(self,
                 encoder: dict[str, any] = None,
                 decoder: dict[str, any] = None,
                 header: dict[str, any] = None):
        super().__init__()
        self.encoder: nn.Module = build_encoder(**encoder)

        feature_info = getattr(self.encoder, 'feature_info', None)
        if feature_info:
            in_channels = feature_info.channels()
            in_strides = feature_info.reduction()
        else:
            in_channels = decoder.pop('in_channels')
            in_strides = decoder.pop('in_strides')

        self.decoder: BaseDecoder = build_decoder(in_channels=in_channels, in_strides=in_strides, **decoder)
        self.header: BaseHeader = build_header(in_channels=self.decoder.out_channels,
                                               in_strides=self.decoder.out_strides, **header)
        self.num_classes = self.header.num_classes

    def forward(self, x: T, target: Optional[dict[str, Any]] = None) -> T | dict[str, T]:
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.header(x, target)
        return x

    def load_weights(self, path: str, unwarp_key: str = 'model.'):
        weights: dict = torch.load(path, map_location='cpu')
        weights: dict = weights['state_dict'] if 'state_dict' in weights.keys() else weights
        weights = {key.replace(unwarp_key, ''): weight for key, weight in weights.items()}
        return self.load_state_dict(weights, strict=True)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
