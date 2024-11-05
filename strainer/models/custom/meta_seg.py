from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F

from strainer.models import ModelBase
from strainer.models.prediction_head.segmentation import CompactSegmentationHead
from strainer.models.layers import ConvBNReLU
from strainer.utils.common import load_weights

from strainer.models.feature_extractor import build_feature_extractor
from strainer.models.feature_decoder import build_feature_decoder


class SegmentationWithMeta(nn.Module):
    def __init__(self, num_classes: int, encoder: dict[str, Any], decoder: dict[str, Any],
                 ):
        super().__init__()
        self.encoder = build_feature_extractor(**encoder)
        self.decoder = build_feature_decoder(**decoder)
