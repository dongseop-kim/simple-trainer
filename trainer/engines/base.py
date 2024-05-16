from abc import abstractmethod
from typing import Any

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from trainer.models import Model


class BaseEngine(LightningModule):
    def __init__(self, model: Model, optimizer=None, scheduler=None,
                 criterion: nn.Module = None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        # each step outputs
        self.train_step_outputs: list[dict[str, torch.Tensor]] = []
        self.validation_step_outputs: list[dict[str, torch.Tensor]] = []
        self.test_step_outputs: list[dict[str, torch.Tensor]] = []
        # self.predict_step_outputs: list[dict[str, torch.Tensor]] = []

    def configure_optimizers(self):
        return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler}

    def aggregate_and_logging(self, outputs: list[dict[str, torch.Tensor]], key: str, prefix: str = None, step: bool = False):
        '''Aggregate outputs from each step and log'''
        loss_per_epoch = torch.stack([x[key] for x in outputs]).mean()
        prefix = '' if prefix is None else prefix + '/'
        self.log(f'{prefix}{key}', loss_per_epoch, on_step=step, on_epoch=not step, prog_bar=True)

    def step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        '''Basic forward step for each task. If you want to customize, override this method'''
        image = batch['image']
        label = batch['label']
        return self._step(image, label)

    def _step(self, image: torch.Tensor, label: torch.Tensor) -> dict[str, torch.Tensor]:
        '''Basic forward step for each task. If you want to customize, override this method'''
        logit, preds = self.model(image, None)
        loss: dict[str, torch.Tensor] = self.criterion(logit, label)
        loss.update({'logit': logit, 'preds': preds})
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        '''Basic training step for each task. If you want to customize, override this method'''
        return self.step(batch)

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        '''Basic validation step for each task. If you want to customize, override this method'''
        return self.step(batch)
