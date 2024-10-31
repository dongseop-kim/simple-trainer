from typing import Optional, Dict
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn as nn
import yaml
from omegaconf import DictConfig

from strainer.utils.common import load_weights


def load_config(config_path: str) -> DictConfig:
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return DictConfig(config)


def instantiate_model(config: DictConfig, num_classes: int):
    model = hydra.utils.instantiate(config.config_model, num_classes=num_classes)
    return model


def instantiate_engine(config: DictConfig, model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       checkpoint: Optional[str] = None,
                       criterion: Optional[nn.Module] = None) -> nn.Module:
    engine = hydra.utils.instantiate(config.config_engine, model=model,
                                     optimizer=optimizer, scheduler=scheduler, criterion=criterion)
    if checkpoint is not None:
        engine = load_weights(engine, checkpoint)
    return engine


def instantiate_key_from_config(config: str | Path | DictConfig, key: str, **kwargs) -> Any:
    if not isinstance(config, DictConfig):
        config = load_config(config)[key]
    else:
        config = config[key]
    return hydra.utils.instantiate(config, **kwargs)
