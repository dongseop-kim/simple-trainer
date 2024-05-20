import timm
import torch.nn as nn


def build_encoder(name: str, pretrained=True, features_only=True, **kwargs) -> timm.models._features.FeatureListNet | nn.Module:
    # NOTE: custom encoder 추가 될 경우 추가하기.
    timm_list = timm.list_models(pretrained=pretrained)
    try:
        return timm.create_model(model_name=name, pretrained=pretrained,
                                 features_only=features_only, **kwargs)
    except:
        raise ValueError(f'Unknown model name: {name}. Available models: {timm_list}')
