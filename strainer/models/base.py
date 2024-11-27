from dataclasses import dataclass
from typing import Any

import torch.nn as nn
from torch import Tensor

from .feature_decoder import BaseFeatureDecoder, build_feature_decoder
from .feature_extractor import build_feature_extractor
from .prediction_head import BasePredictionHead, build_prediction_head


@dataclass
class FeatureInfo:
    """데이터 클래스로 feature 정보를 관리"""
    channels: list[int]
    scales: list[int]


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
            input_channels = config_decoder.pop('input_scales')
            input_scales = config_decoder.pop('input_channels')

        self.feature_decoder: BaseFeatureDecoder = build_feature_decoder(input_channels=input_channels, input_scales=input_scales,
                                                                         **config_decoder)

        self.prediction_head: BasePredictionHead = build_prediction_head(input_channels=self.feature_decoder.output_channels,
                                                                         input_scales=self.feature_decoder.output_scales,
                                                                         **config_header)
        self.num_classes = self.prediction_head.num_classes

    def _get_feature_info(self, config: dict[str, Any]) -> FeatureInfo:
        """Feature 정보 추출"""
        if hasattr(self.feature_extractor, 'feature_info'):
            feature_info = self.feature_extractor.feature_info
            return FeatureInfo(channels=feature_info.channels(), scales=feature_info.reduction())
        return FeatureInfo(channels=config.pop('in_channels'), scales=config.pop('in_strides'))

    def forward(self, images: Tensor, targets: dict[str, Any] | None = None) -> dict[str, Tensor]:
        features = self.feature_extractor(images)
        decoded_features = self.feature_decoder(features)
        predictions = self.prediction_head(decoded_features, targets)
        return predictions

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        return [param for param in self.parameters() if param.requires_grad]
