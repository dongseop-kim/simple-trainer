from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn as nn
import yaml
from omegaconf import DictConfig


def load_config(config_path: str) -> DictConfig:
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return DictConfig(config)


def instantiate_model(config: DictConfig, num_classes: int):
    model = hydra.utils.instantiate(config.config_model, num_classes=num_classes)
    return model


def instantiate_engine(config: DictConfig, model, optimizer=None, scheduler=None, checkpoint=None):
    engine: nn.Module = hydra.utils.instantiate(config.config_engine, model=model,
                                                optimizer=optimizer, scheduler=scheduler)
    if not checkpoint:
        return engine
    checkpoint = torch.load(checkpoint, map_location='cpu')
    checkpoint = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    engine.load_state_dict(checkpoint)
    return engine


def instantiate_key_from_config(config: str | Path | DictConfig,
                                key: str, **kwargs) -> Any:
    """
    Instantiates object from config file.

    Args:
        config (Union[str, Path, DictConfig]): Path to yaml file or DictConfig object.
        key (str): Key in config file. e.g. (datamodule, model, engine)
        **kwargs: Additional arguments passed to instantiate function.
    """
    if not isinstance(config, DictConfig):
        config = load_config(config)[key]
    else:
        config = config[key]
    return hydra.utils.instantiate(config, **kwargs)
