from typing import Any

import torch.nn as nn
from torch import Tensor

from .feature_decoder import BaseFeatureDecoder, build_feature_decoder
from .feature_extractor import build_feature_extractor
from .prediction_head import BasePredictionHead, build_prediction_head


class ModelBase(nn.Module):
    def __init__(self,
                 config_encoder: dict[str, any],
                 config_decoder: dict[str, any],
                 config_header: dict[str, any]):
        super().__init__()
        self.feature_extractor: nn.Module = build_feature_extractor(**config_encoder)

        feature_info = getattr(self.feature_extractor, 'feature_info', None)
        if feature_info:
            input_channels = feature_info.channels()
            input_scales = feature_info.reduction()
        else:
            input_channels = config_decoder.pop('in_channels')
            input_scales = config_decoder.pop('in_strides')

        self.feature_decoder: BaseFeatureDecoder = build_feature_decoder(in_channels=input_channels, in_strides=input_scales,
                                                                         **config_decoder)

        self.prediction_head: BasePredictionHead = build_prediction_head(input_channels=self.feature_decoder.output_channels,
                                                                         input_scales=self.feature_decoder.output_scales,
                                                                         **config_header)
        self.num_classes = self.header.num_classes

    def forward(self, images: Tensor, targets: dict[str, Any] | None = None) -> dict[str, Tensor]:
        features = self.feature_extractor(images)
        decoded_features = self.feature_decoder(features)
        predictions = self.prediction_head(decoded_features, targets)
        return predictions

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        return [param for param in self.parameters() if param.requires_grad]
