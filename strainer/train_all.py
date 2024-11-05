import shutil
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (LearningRateMonitor, RichModelSummary,
                                         RichProgressBar)
from univdt.datamodules import BaseDataModule

from strainer.models import ModelBase
from strainer.utils.config import load_config

torch.set_float32_matmul_precision('medium')  # 또는 'high'


def train_all(path: str, debug: bool = False):
    config: DictConfig = load_config(path)
    # copy to save_dir
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, save_dir / 'config.yaml')

    '''Build DataModule'''
    datamodule: BaseDataModule = hydra.utils.instantiate(config.config_datamodule)

    '''Build Model'''
    model: ModelBase = hydra.utils.instantiate(config.config_model)

    '''Build Optimizer & Scheduler & Criterion'''
    optimizer = hydra.utils.instantiate(config.config_optimizer, params=model.parameters())
    scheduler = hydra.utils.instantiate(config.config_scheduler, optimizer=optimizer,
                                        iter_per_epoch=len(datamodule.train_dataloader()))
    criterion = None
    if "config_criterion" in config:
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

    # tmp code
    # engine.model.freeze_encoder()
    # initial_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
    #                                     lr=0.01, momentum=0.9)
    # initial_scheduler = torch.optim.lr_scheduler.ConstantLR(initial_optimizer, factor=1.0,
    #                                                         total_iters=0)
    # initial_engine = hydra.utils.instantiate(config.config_engine,
    #                                          model=model,
    #                                          optimizer=initial_optimizer,
    #                                          scheduler=initial_scheduler,
    #                                          criterion=criterion)
    # initial_trainer: Trainer = hydra.utils.instantiate(config.config_trainer, max_epochs=5, check_val_every_n_epoch=5)
    # initial_trainer.fit(model=initial_engine, datamodule=datamodule)
    # engine.model.unfreeze_encoder()
    # end tmp code

    '''Build Trainer'''
    trainer: Trainer = hydra.utils.instantiate(config.config_trainer, callbacks=callbacks, logger=logger)
    trainer.fit(model=engine, datamodule=datamodule)
