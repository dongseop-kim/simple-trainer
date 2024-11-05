import timm
import torch.nn as nn

from timm.models._features import FeatureListNet

# Custom feature extractors registry
custom_feature_extractors: dict[str, nn.Module] = {'custom_resnet': None,  # CustomResNet,
                                                   }


def build_feature_extractor(model_name: str, pretrained: bool = True,
                            features_only: bool = True,
                            **kwargs) -> FeatureListNet | nn.Module:
    """
    Build a feature extractor module based on the given model name and parameters.

    This builder prioritizes the use of timm (PyTorch Image Models) library for its extensive
    collection of pre-implemented and pre-trained models. It also supports custom feature 
    extractors registered in custom_feature_extractors.

    Args:
        model_name (str): Name of the model architecture. Can be either a timm model or
            a custom model registered in custom_feature_extractors.
        pretrained (bool, optional): Whether to load pretrained weights. Only applicable
            for timm models. Defaults to True.
        features_only (bool, optional): If True, the model will return intermediate feature
            maps instead of classification outputs. Only applicable for timm models. 
            Defaults to True.
        **kwargs: Additional arguments to pass to the model constructor.

    Returns:
        FeatureListNet | nn.Module: The constructed feature extractor model.
            Returns FeatureListNet when using timm model with features_only=True,
            otherwise returns nn.Module.

    Raises:
        ValueError: If model_name is not found in either timm models or custom_feature_extractors.
        RuntimeError: If model creation fails for any reason.

    Examples:
        >>> # Using a timm model
        >>> extractor = build_feature_extractor('resnet50', pretrained=True)
        >>> # Using a custom model
        >>> extractor = build_feature_extractor('custom_resnet', pretrained=False)

    Notes:
        - For timm models, feature information can be accessed via the feature_info attribute
        - Supported timm models can be found using timm.list_models()
        - Custom models should implement the same interface as timm models for consistency
    """

    if model_name in custom_feature_extractors:
        if custom_feature_extractors[model_name] is None:
            raise NotImplementedError(f"Custom model '{model_name}' is registered but not implemented yet")
        try:
            return custom_feature_extractors[model_name](**kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to create custom model '{model_name}': {str(e)}")

    # try timm models
    available_models = timm.list_models(pretrained=pretrained)  # get only models with pretrained weights
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' not found in timm registry or custom models.\n"
                         f"Available timm models: {', '.join(sorted(available_models))}\n"
                         f"Available custom models: {', '.join(sorted(custom_feature_extractors.keys()))}")
    try:
        return timm.create_model(model_name=model_name, pretrained=pretrained,
                                 features_only=features_only, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to create timm model '{model_name}': {str(e)}")
