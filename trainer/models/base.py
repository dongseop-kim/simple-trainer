from typing import Any, Dict, Optional

import timm
import torch
import torch.nn as nn

from .encoder import build_encoder
from .decoder import BaseDecoder, build_decoder
from .header import BaseHeader, build_header


class Model(nn.Module):
    def __init__(self,
                 num_classes: int,
                 encoder: Dict[str, any] = None,
                 decoder: Dict[str, any] = None,
                 header: Dict[str, any] = None):
        super().__init__()
        self.num_classes = num_classes

        self.encoder: nn.Module = build_encoder(**encoder)
        self.decoder: BaseDecoder = build_decoder(in_channels=self.encoder.feature_info.channels(),
                                                  in_strides=self.encoder.feature_info.reduction(),
                                                  **decoder)
        self.header = build_header(num_classes=num_classes, in_channels=self.decoder.out_channels,
                                   in_strides=self.decoder.out_strides, **header)

    def forward(self, x: torch.Tensor,
                target: Optional[dict[str, Any]] = None) -> torch.Tensor | Dict[str, torch.Tensor]:
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.header(x, target)
        return x

    def load_weights(self, path: str, unwarp_key: str = 'model.'):
        weights: Dict = torch.load(path, map_location='cpu')
        weights: Dict = weights['state_dict'] if 'state_dict' in weights.keys() else weights
        weights = {key.replace(unwarp_key, ''): weight for key, weight in weights.items()}
        return self.load_state_dict(weights, strict=True)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
