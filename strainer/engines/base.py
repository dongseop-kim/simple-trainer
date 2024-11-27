import logging
from typing import Any

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor

from strainer.models import ModelBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_ABBREVIATIONS = {'binaryf1score': 'f1',
                  'binaryrecall': 'sens',
                  'binaryprecision': 'prec',
                  'binaryspecificity': 'spec',
                  'binaryaccuracy': 'acc',
                  'meanabsoluteerror': 'mae',
                  'meansquarederror': 'mse',
                  }

# https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#hooks


class BaseEngine(LightningModule):
    def __init__(self, model: ModelBase, optimizer=None, scheduler=None, criterion: nn.Module = None, **kwargs):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        # each step outputs
        self.train_step_outputs: list[dict[str, Tensor]] = []
        self.validation_step_outputs: list[dict[str, Tensor]] = []
        self.test_step_outputs: list[dict[str, Tensor]] = []

    def configure_optimizers(self):
        return {'optimizer': self.optimizer,
                'lr_scheduler': {'scheduler': self.scheduler, 'interval': 'step'}}

    def aggregate_and_logging(self, outputs: list[dict[str, Tensor]], key: str, prefix: str = None, step: bool = False):
        '''Aggregate outputs from each step and log'''
        loss_per_epoch = torch.stack([x[key] for x in outputs]).mean()
        prefix = '' if prefix is None else prefix + '/'
        self.log(f'{prefix}{key}', loss_per_epoch, on_step=step, on_epoch=not step, prog_bar=True)

    def step(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        '''Basic forward step for each task. If you want to customize, override this method'''
        image = batch['image']
        label = batch['label']
        return self._step(image, label)

    def _step(self, image: Tensor, label: Tensor) -> dict[str, Tensor]:
        '''Basic forward step for each task. If you want to customize, override this method'''
        output: dict[str, Tensor] = self.model(image, None)

        logits: Tensor = output['logits']

        loss: dict[str, Tensor] = self.criterion(logits, label)
        loss.update(output)
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        '''Basic training step for each task. If you want to customize, override this method'''
        return self.step(batch)

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        '''Basic validation step for each task. If you want to customize, override this method'''
        return self.step(batch)
