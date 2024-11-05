from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F

from strainer.models import ModelBase
from strainer.models.prediction_head.segmentation import CompactSegmentationHead
from strainer.models.layers import ConvBNReLU
from strainer.utils.common import load_weights


class ChannelAttention(nn.Module):
    def __init__(self, img_dim, metadata_dim, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(img_dim + metadata_dim, (img_dim + metadata_dim) // reduction, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear((img_dim + metadata_dim) // reduction, img_dim, bias=False),
                                nn.Sigmoid())

    def forward(self, img_features: torch.Tensor, metadata_features: torch.Tensor):
        b, c, _, _ = img_features.shape
        y = self.avg_pool(img_features).view(b, c)
        y = torch.cat([y, metadata_features], dim=1)
        y = self.fc(y).view(b, c, 1, 1)
        return img_features * y.expand_as(img_features)


class CrossAttention(nn.Module):
    def __init__(self, img_dim=256, metadata_dim=512, num_heads=8):
        super().__init__()
        self.img_dim = img_dim
        self.metadata_dim = metadata_dim
        self.num_heads = num_heads
        self.head_dim = metadata_dim // num_heads

        self.queries = nn.Linear(img_dim, metadata_dim)
        self.keys = nn.Linear(metadata_dim, metadata_dim)
        self.values = nn.Linear(metadata_dim, metadata_dim)

        self.fc_out = nn.Linear(metadata_dim, img_dim)

    def forward(self, img_features, metadata_features):
        N, C, H, W = img_features.shape

        # Reshape image features to (N, H*W, C)
        img_features = img_features.view(N, C, H*W).permute(0, 2, 1)

        # Project queries, keys, and values
        Q = self.queries(img_features)
        K = self.keys(metadata_features.unsqueeze(1))
        V = self.values(metadata_features.unsqueeze(1))

        # Reshape for multi-head attention
        Q = Q.view(N, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(N, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(N, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute attention scores
        energy = torch.einsum("nhqd,nhkd->nhqk", [Q, K]) / (self.head_dim ** 0.5)
        attention = F.softmax(energy, dim=-1)

        # Apply attention to values
        out = torch.einsum("nhql,nhld->nqhd", [attention, V])
        out = out.permute(0, 2, 1, 3).contiguous().view(N, H*W, self.metadata_dim)

        # Project back to image feature dimension
        out = self.fc_out(out)

        # Reshape to original image feature shape
        out = out.view(N, H, W, C).permute(0, 3, 1, 2)

        return out


class MetaSegmentationModel(nn.Module):
    def __init__(self, num_classes: int,
                 feature_encoder: dict[str, Any], meta_encoder: dict[str, Any],
                 weight_feature_encoder: str, weight_meta_encoder: str,
                 img_dim: int = 256, metadata_dim: int = 512,
                 mode: str = 'cross_attention'):
        super().__init__()
        self.feature_encoder = ModelBase(**feature_encoder)
        self.feature_encoder.load_state_dict(load_weights(weight_feature_encoder, 'model.'))

        self.meta_encoder = ModelBase(**meta_encoder)
        self.meta_encoder.load_state_dict(load_weights(weight_meta_encoder, 'model.'))

        if mode == 'cross_attention':
            self.attention = CrossAttention(img_dim, metadata_dim)
        elif mode == 'channel_attention':
            self.attention = ChannelAttention(img_dim, metadata_dim)
        else:
            raise ValueError(f"Invalid attention mode: {mode}")

        self.header = CompactSegmentationHead(num_classes, in_channels=img_dim, in_strides=8,
                                              return_logits=True, init_prob=0.01)

    def _check_input(self, x: dict[str, Any] | torch.Tensor):
        if isinstance(x, dict):
            assert 'image' in x
            return x['image']
        elif isinstance(x, torch.Tensor):
            return x
        else:
            raise ValueError(f"Invalid input type: {type(x)}")

    def forward(self, x, *args, **kwargs):
        x = self._check_input(x)
        feature_image = self._get_feature_image(x)
        feature_meta = self._get_feature_meta(x)

        feature_image = self.attention(feature_image, feature_meta)  # (N, 256, H/32, W/32)

        out: tuple[torch.Tensor] = self.header(feature_image)

        return out

    def _get_feature_image(self, x):
        x = self.feature_encoder.feature_extractor(x)
        x = self.feature_encoder.feature_decoder(x)
        return x  # output shape: (N, 256, H/32, W/32)

    def _get_feature_meta(self, x):
        # resize to 384
        x = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=True)
        x = self.meta_encoder.feature_extractor(x)
        x = self.meta_encoder.feature_decoder(x)
        if isinstance(x, tuple) or isinstance(x, list):
            assert len(x) == 1
            x = x[0]

        # apply global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))  # N x 512 x 1 x 1
        x = x.view(x.size(0), -1)  # N x 512
        return x  # output shape: (N, 512)

    def freeze_encoder(self, feature_encoder: bool = True, meta_encoder: bool = True):
        if feature_encoder:
            for param in self.feature_encoder.parameters():
                param.requires_grad = False
            self.feature_encoder.eval()
        if meta_encoder:
            for param in self.meta_encoder.parameters():
                param.requires_grad = False
            self.meta_encoder.eval()

    def unfreeze_encoder(self):
        for param in self.feature_encoder.feature_extractor.parameters():
            param.requires_grad = True
        for param in self.feature_encoder.feature_decoder.parameters():
            param.requires_grad = True
        for param in self.meta_encoder.parameters():
            param.requires_grad = True

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]
