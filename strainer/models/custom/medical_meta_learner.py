import torch.nn as nn
from torch import Tensor as T

from strainer.models import ModelBase
from strainer.models.prediction_head.linear import SingleConvHeader
from strainer.models.layers import ConvBNReLU


class MedicalMetaModel(ModelBase):
    def __init__(self, encoder: dict[str, any], decoder: dict[str, any]):
        super().__init__(config_encoder=encoder, config_decoder=decoder, config_header={'name': 'identity',
                                                                                        'num_classes': -1})
        in_channels: int | list[int] = self.feature_decoder.out_channels
        assert len(in_channels) == 1, "Only support single input channel."
        in_channels = in_channels[0] if isinstance(in_channels, list) else in_channels

        self.conv = ConvBNReLU(in_channels, 512, kernel_size=3, stride=1, padding=1)

        self.header_age = nn.Sequential(ConvBNReLU(512, 512, 3, 1, 1),
                                        SingleConvHeader(num_classes=96, in_channels=512, in_strides=-1, pool='avg',
                                                         return_logits=True))
        self.header_gender = nn.Sequential(ConvBNReLU(512, 512, 3, 1, 1),
                                           SingleConvHeader(num_classes=2, in_channels=512, in_strides=-1, pool='avg',
                                                            return_logits=True))
        self.softmax = nn.Softmax2d()

    def forward(self, x: T, target=None) -> dict[str, T]:
        features: T = super().forward(x, target)[0]
        features = self.conv(features)

        logit_age, pred_age = self.header_age(features)
        logits_gender, _ = self.header_gender(features)
        pred_gender = self.softmax(logits_gender)

        N, C = logit_age.shape[:2]
        logit_age = logit_age.view(N, C)
        pred_age = pred_age.view(N, C)

        N, C = logits_gender.shape[:2]
        logits_gender = logits_gender.view(N, C)
        pred_gender = pred_gender.view(N, C)

        return {'logit_age': logit_age, 'pred_age': pred_age,
                'logit_gender': logits_gender, 'pred_gender': pred_gender}
