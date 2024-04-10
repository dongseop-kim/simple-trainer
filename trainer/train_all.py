import argparse
import shutil
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         RichModelSummary, RichProgressBar)

# from trainer.datamodule import TBDataModule
from trainer.models import Model
# from trainer.utils.common import load_config


def train_all(config: DictConfig):
    '''Build DataModule'''
    datamodule: LightningDataModule = TBDataModule(**config['config_datamodule'])
    datamodule.prepare_data()
    datamodule.setup('fit')
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    print(len(train_dataloader), train_dataloader.batch_size, len(train_dataloader.dataset))
    print(len(val_dataloader), val_dataloader.batch_size, len(val_dataloader.dataset))

    '''Build Model'''
    num_classes: int = 1

    model: Model = hydra.utils.instantiate(config.config_model, num_classes=num_classes)

    '''Build Optimizer & Scheduler'''
    optimizer = hydra.utils.instantiate(config.config_optimizer, params=model.parameters())
    scheduler = hydra.utils.instantiate(config.config_scheduler, optimizer=optimizer)

    '''Build Engine'''
    engine = hydra.utils.instantiate(config.config_engine, model=model, optimizer=optimizer, scheduler=scheduler)

    '''Build Logger'''
    logger = hydra.utils.instantiate(config.config_logger)

    '''Build Callbacks'''
    cb_model_summary = RichModelSummary(max_depth=1)
    cb_progress_bar = RichProgressBar()
    cb_lr_monitor = LearningRateMonitor(logging_interval='step')
    # 나중에 config로 옮기기
    cb_model_checkpoint = ModelCheckpoint(monitor='val/f1score', mode='max',
                                          save_top_k=3, save_last=False,
                                          dirpath=Path(config.save_dir) / 'checkpoints/',
                                          filename='epoch_{epoch:03d}')

    '''Build Trainer'''
    trainer: Trainer = hydra.utils.instantiate(config.config_trainer,
                                               callbacks=[cb_model_summary, cb_progress_bar,
                                                          cb_lr_monitor, cb_model_checkpoint],
                                               logger=logger)
    trainer.fit(model=engine, datamodule=datamodule)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    # copy to save_dir
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, save_dir / 'config.yaml')
    train_all(config)
