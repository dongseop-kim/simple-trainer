import logging
from typing import Any

import torch
from torch import Tensor
from torchmetrics import (F1Score, MetricCollection, Precision, Recall,
                          Specificity)

from strainer.engines.base import _ABBREVIATIONS, BaseEngine
from strainer.models import ModelBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BinarySegmentation(BaseEngine):
    def __init__(self, model: ModelBase, optimizer=None, scheduler=None, criterion=None, **kwargs):
        super().__init__(model, optimizer, scheduler, criterion, **kwargs)

        self.meter_train = MetricCollection([F1Score(task='binary'), Recall(task='binary'),
                                             Precision(task='binary'), Specificity(task='binary')])
        self.meter_valid = MetricCollection([F1Score(task='binary'), Recall(task='binary'),
                                             Precision(task='binary'), Specificity(task='binary')])

    def step(self, batch: dict[str, Any]) -> dict[str, Any]:
        output: dict[str, Tensor] = self.model(batch['image'], None)
        logits: Tensor = output['logits']
        loss: dict[str, Tensor] = self.criterion(logits, batch['mask'])
        loss.update(output)
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
