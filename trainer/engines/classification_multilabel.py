from typing import Any

import torch
import torchmetrics
from torchmetrics import MetricCollection
from torchmetrics.classification import ConfusionMatrix, MulticlassAccuracy

from trainer.engines.base import BaseEngine
from trainer.models import Model


_EPS = 1e-7

criterions = {'ce': torch.nn.CrossEntropyLoss,
              'bce': torch.nn.BCEWithLogitsLoss}


class ClassificationEngine(BaseEngine):
    def __init__(self, model: Model, optimizer=None, scheduler=None, criterion=None):
        super().__init__(model, optimizer, scheduler)

        self.criterion = criterions[criterion]()

        # hard-coded metrics
        # TODO: 이 부분 개선하기 -> 인자로 처리 가능하도록
        self.meter_train = MulticlassAccuracy(10, 3)  # 10 classes, top3
        self.meter_valid = MulticlassAccuracy(10, 1)  # 10 classes, top1

    def step(self, batch: dict[str, Any]) -> dict[str, Any]:
        logit, preds = self.model(batch['image'], None)
        loss = self.criterion(logit.squeeze(), batch['label'].squeeze())
        return {'loss': loss, 'logit': logit, 'preds': preds}

    ''' ====================== '''
    ''' ===== TRAINING ===== '''
    ''' ====================== '''

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        return self.step(batch)

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int):
        self.log('train/loss', outputs['loss'], on_step=True, on_epoch=False, prog_bar=True)
        self.train_step_outputs.append(outputs)  # save outputs

        self.meter_train(outputs['preds'].squeeze(), batch['label'].squeeze())
        scores = self.meter_train.compute()
        self.log('train/top3', scores, on_step=True, on_epoch=False, prog_bar=True)

    def on_train_epoch_end(self):
        self.aggregate_and_logging(self.train_step_outputs, 'loss', prefix='train', step=False)
        self.train_step_outputs.clear()
        self.meter_train.reset()

    ''' ====================== '''
    ''' ===== VALIDATION ===== '''
    ''' ====================== '''

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        return self.step(batch)

    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int):
        self.validation_step_outputs.append(outputs)
        self.meter_valid(outputs['preds'].squeeze(), batch['label'].squeeze())

    def on_validation_epoch_end(self):
        scores = self.meter_valid.compute()
        self.log('val/top1', scores, on_step=False, on_epoch=True, prog_bar=True)
        self.meter_valid.reset()
        self.aggregate_and_logging(self.validation_step_outputs, 'loss', prefix='val', step=False)
        self.validation_step_outputs.clear()

    ''' ====================== '''
    ''' ====== PREDICT  ====== '''
    ''' ====================== '''

    def predict_step(self, batch: dict[str, Any], batch_idx: int):
        # TODO: implement predict_step
        pass
