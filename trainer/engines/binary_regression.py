from typing import Any

import torch
from torch import Tensor as T
from torchmetrics.classification import ConfusionMatrix

from trainer.engines.base import BaseEngine
from trainer.models import Model

_EPS = 1e-7

criterions = {'ce': torch.nn.CrossEntropyLoss,
              'bce': torch.nn.BCEWithLogitsLoss}


class BinaryRegressionEngine(BaseEngine):
    def __init__(self, model: Model, optimizer=None, scheduler=None, criterion=None, metric=None):
        super().__init__(model, optimizer, scheduler)

        self.criterion = criterions[criterion]()

        # hard-coded metrics
        # TODO: 이 부분 개선하기 -> 인자로 처리 가능하도록
        # metrics
        self.meter_train = ConfusionMatrix(task='binary')
        self.meter_valid = ConfusionMatrix(task='binary')

    def step(self, batch: dict[str, Any]) -> dict[str, Any]:
        logit, preds = self.model(batch['image'], None)
        loss = self.criterion(logit.squeeze(), batch['mask'].squeeze().float())
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
        scores: dict[str, T] = self.compute_confusion_matrix(self.meter_train)
        scores = {f'train/{k}': v for k, v in scores.items()}
        self.log_dict(scores, on_step=True, on_epoch=False, prog_bar=True)

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
        scores = self.compute_confusion_matrix(self.meter_valid)
        scores = {f'val/{k}': v for k, v in scores.items()}
        self.log_dict(scores, on_step=False, on_epoch=True, prog_bar=True)
        self.meter_valid.reset()
        self.aggregate_and_logging(self.validation_step_outputs, 'loss', prefix='val', step=False)
        self.validation_step_outputs.clear()

    ''' ====================== '''
    ''' ====== PREDICT  ====== '''
    ''' ====================== '''

    def predict_step(self, batch: dict[str, Any], batch_idx: int):
        # TODO: implement predict_step
        pass

    def compute_confusion_matrix(self, meter: ConfusionMatrix):
        '''Compute confusion matrix'''
        _eps = 1e-7
        confusion_matrix: T = meter.compute()
        tn, fp, fn, tp = confusion_matrix.view(-1)
        accuracy = (tp + tn) / (tp + tn + fp + fn + _eps)
        precision = tp / (tp + fp + _eps)
        sensitivity = tp / (tp + fn + _eps)
        specificity = tn / (tn + fp + _eps)
        f1score = 2 * (precision * sensitivity) / (precision + sensitivity + _eps)
        return {'acc': accuracy, 'prec': precision, 'sens': sensitivity,
                'spec': specificity, 'f1': f1score}
