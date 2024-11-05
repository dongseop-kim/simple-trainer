import logging
from typing import Any

import torch
from torch import Tensor as T
from torch import nn
from torchmetrics import (F1Score, MetricCollection, Precision, Recall,
                          Specificity)

from strainer.engines.base import BaseEngine
from strainer.models import ModelBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_ABBREVIATIONS = {'binaryf1score': 'f1',
                  'binaryrecall': 'sens',
                  'binaryprecision': 'prec',
                  'binaryspecificity': 'spec',
                  }


class BinarySegmentation(BaseEngine):
    def __init__(self, model: ModelBase, optimizer=None, scheduler=None, criterion=None,
                 freeze_encoder: bool = False, **kwargs):
        super().__init__(model, optimizer, scheduler, criterion, **kwargs)

        self.meter_train = MetricCollection([F1Score(task='binary'), Recall(task='binary'),
                                             Precision(task='binary'), Specificity(task='binary')])
        self.meter_valid = MetricCollection([F1Score(task='binary'), Recall(task='binary'),
                                             Precision(task='binary'), Specificity(task='binary')])

        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            self.model.freeze_encoder()

    def step(self, batch: dict[str, Any]) -> dict[str, Any]:
        logit, preds = self.model(batch['image'], None)  # N x 1 x H x W
        loss: dict = self.criterion(logit, batch['mask'])
        loss.update({'logit': logit, 'preds': preds})
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        return self.step(batch)

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        return self.step(batch)

    def _get_scores(self, meter: MetricCollection, prefix: str) -> dict[str, Any]:
        scores: dict[str, Any] = meter.compute()
        scores = {_ABBREVIATIONS.get(k.lower(), k): v for k, v in scores.items()}
        scores = {f'{prefix}/{k}': v for k, v in scores.items()}
        return scores

    def _process_batch_end(self, outputs, batch: dict[str, Any], meter: MetricCollection) -> dict[str, Any]:
        preds: torch.Tensor = outputs['preds'].detach()
        target: torch.Tensor = batch['mask'].detach()
        loss: torch.Tensor = outputs['loss'].detach()

        preds = preds.squeeze().flatten(start_dim=1)
        target = target.squeeze().flatten(start_dim=1)
        preds = preds.max(dim=1).values
        target = target.max(dim=1).values
        meter.update(preds, target)

        return {'loss': loss}

    def on_train_epoch_start(self):
        if self.freeze_encoder:
            self.model.freeze_encoder()

    def on_train_batch_end(self, outputs, batch: dict[str, Any], batch_idx: int):
        batch_output = self._process_batch_end(outputs, batch, self.meter_train)
        self.train_step_outputs.append(batch_output)

        scores = self._get_scores(self.meter_train, 'train')
        scores.update({'train/loss': batch_output['loss']})
        self.log_dict(scores, on_step=True, on_epoch=False, prog_bar=True)

    def on_validation_batch_end(self, outputs, batch: dict[str, Any], batch_idx: int):
        batch_output = self._process_batch_end(outputs, batch, self.meter_valid)
        self.validation_step_outputs.append(batch_output)

    def on_train_epoch_end(self):
        self.aggregate_and_logging(self.train_step_outputs, 'loss', prefix='train', step=False)
        self.train_step_outputs.clear()
        self.meter_train.reset()

    def on_validation_epoch_end(self):
        scores = self._get_scores(self.meter_valid, 'val')
        self.log_dict(scores, on_step=False, on_epoch=True, prog_bar=True)
        self.aggregate_and_logging(self.validation_step_outputs, 'loss', prefix='val', step=False)
        self.validation_step_outputs.clear()
        self.meter_valid.reset()
