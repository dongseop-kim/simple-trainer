import shutil
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (LearningRateMonitor, RichModelSummary,
                                         RichProgressBar)
from univdt.datamodules import BaseDataModule

from trainer.models import Model
from trainer.utils.config import load_config


def train_all(path: str, debug: bool = False):
    config: DictConfig = load_config(path)
    # copy to save_dir
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, save_dir / 'config.yaml')

    '''Build DataModule'''
    datamodule: BaseDataModule = hydra.utils.instantiate(config.config_datamodule)

    '''Build Model'''
    model: Model = hydra.utils.instantiate(config.config_model)

    '''Build Optimizer & Scheduler & Criterion'''
    optimizer = hydra.utils.instantiate(config.config_optimizer, params=model.parameters())
    scheduler = hydra.utils.instantiate(config.config_scheduler, optimizer=optimizer)
    criterion = hydra.utils.instantiate(config.config_criterion)

    '''Build Engine'''
    engine = hydra.utils.instantiate(config.config_engine, model=model,
                                     optimizer=optimizer, scheduler=scheduler, criterion=criterion)

    '''Build Logger'''
    logger = hydra.utils.instantiate(config.config_logger)

    '''Build Callbacks'''
    callbacks: list[pl.Callback] = list()
    if "config_callbacks" in config:
        callbacks = [hydra.utils.instantiate(_conf) for _conf in config.config_callbacks.values()]
    callbacks.append(RichModelSummary(max_depth=1))
    callbacks.append(RichProgressBar())
    callbacks.append(LearningRateMonitor(logging_interval='step'))

    if debug:
        return config, datamodule, model, optimizer, scheduler, criterion, engine, logger, callbacks

    '''Build Trainer'''
    trainer: Trainer = hydra.utils.instantiate(config.config_trainer, callbacks=callbacks, logger=logger)
    trainer.fit(model=engine, datamodule=datamodule)
