from typing import Any

import torch
from torch import Tensor as T
from torchmetrics import (AUROC, F1Score, MetricCollection, Precision, Recall,
                          Specificity)
from torchmetrics.classification import ConfusionMatrix

from strainer.engines.base import BaseEngine
from strainer.models import Model


class BinarySegmentation(BaseEngine):
    def __init__(self, model: Model, optimizer=None, scheduler=None, criterion=None):
        super().__init__(model, optimizer, scheduler, criterion)

        self.meter_train = MetricCollection([F1Score(task='binary', threshold=0.5)])
        self.meter_valid = MetricCollection([F1Score(task='binary', threshold=0.5)])

    def step(self, batch: dict[str, Any]) -> dict[str, Any]:
        logit, preds = self.model(batch['image'], None)
        loss: dict = self.criterion(logit, batch['mask'])
        loss.update({'logit': logit, 'preds': preds})
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        return self.step(batch)

    def on_train_batch_end(self, outputs, batch: dict[str, Any], batch_idx: int):
        self.log('train/loss', outputs['loss'], on_step=True, on_epoch=False, prog_bar=True)
        self.train_step_outputs.append(outputs)  # save outputs

        self.meter_train.update(outputs['preds'].squeeze(), batch['mask'].squeeze())
        scores = self.meter_train.compute()
        scores = {f'train/{k}': v for k, v in scores.items()}
        self.log_dict(scores, on_step=True, on_epoch=False, prog_bar=True)

    def on_train_epoch_end(self):
        self.aggregate_and_logging(self.train_step_outputs, 'loss', prefix='train', step=False)
        self.train_step_outputs.clear()
        self.meter_train.reset()

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        return self.step(batch)

    def on_validation_batch_end(self, outputs, batch: dict[str, Any], batch_idx: int):
        self.validation_step_outputs.append(outputs)
        self.meter_valid.update(outputs['preds'].squeeze(), batch['mask'].squeeze())

    def on_validation_epoch_end(self):
        scores = self.meter_valid.compute()
        scores = {f'val/{k}': v for k, v in scores.items()}
        self.log_dict(scores, on_step=False, on_epoch=True, prog_bar=True)
        self.meter_valid.reset()
        self.aggregate_and_logging(self.validation_step_outputs, 'loss', prefix='val', step=False)
        self.validation_step_outputs.clear()
