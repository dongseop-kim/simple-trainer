from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from strainer.criteria.age import CoralLoss
from strainer.criteria.ce import CrossEntropy
from strainer.engines.base import BaseEngine
from strainer.metrics.confusion_matrix import ConfusionMatrix
from strainer.metrics.mae import MAE
from strainer.metrics.mse import RMSE
from strainer.models.custom.medical_meta_learner import MedicalMetaModel
from torch import Tensor as T
from torchmetrics import MetricCollection


class MedicalMetaLoss(nn.Module):
    def __init__(self, weight_age: float = 1.0, weight_gender: float = 0.5):
        super().__init__()
        self.weight_age = weight_age
        self.weight_gender = weight_gender
        self.criterion_age = CoralLoss(num_classes=96)
        self.criterion_gender = CrossEntropy(ignore_index=255, num_classes=2)

    def forward(self, x: dict[str, T], target: dict[str, T]) -> dict[str, T]:
        loss_age = self.criterion_age(x['logit_age'], target['age'])['loss']
        loss_gender = self.criterion_gender(x['logit_gender'], target['gender'])['loss']
        loss_age = loss_age * self.weight_age
        loss_gender = loss_gender * self.weight_gender
        loss = loss_age + loss_gender
        return {'loss': loss, 'loss_age': loss_age, 'loss_gender': loss_gender}


class MedicalMetaLearner(BaseEngine):
    '''
    Medical MetaLearner Engine. This engine is used for medical meta-learning tasks. e.g. age, gender, view position, etc.
    '''

    def __init__(self, model: MedicalMetaModel, optimizer=None, scheduler=None, criterion=None):
        super().__init__(model, optimizer, scheduler, criterion)

        # for age estimation
        self.meter_age_train = MetricCollection([MAE(), RMSE()], prefix='train/')
        self.meter_age_valid = MetricCollection([MAE(), RMSE()], prefix='valid/')

        # for gender estimation
        self.meter_gender_train = ConfusionMatrix(num_labels=2, prefix='train/')
        self.meter_gender_valid = ConfusionMatrix(num_labels=2, prefix='valid/')

    def step(self, batch: dict[str, Any]) -> dict[str, Any]:
        '''Override the step method to customize the forward step'''
        batch['gender'] = self._preprocess_gender(batch['gender'])
        gender = T(gender).long().to(batch['label'].device)

        output: dict[str, T] = self.model(batch['image'], None)
        loss: dict[str, T] = self.criterion(output, batch)
        output.update(loss)
        return output

    def on_train_batch_end(self, outputs: dict[str, Any], batch: Any, batch_idx: int):
        outputs['train/loss'] = outputs.pop('loss')
        outputs['train/loss_age'] = outputs.pop('loss_age')
        outputs['train/loss_gender'] = outputs.pop('loss_gender')
        self.train_step_outputs.append(outputs)

        # update metrics
        self.meter_age_train(outputs['pred_age'], batch['age'])
        self.meter_gender_train(outputs['pred_gender'], F.one_hot(batch['gender']))

        # logging
        results: dict[str, T] = self.meter_age_train.compute()
        results.update(self.meter_gender_train.compute())
        results = {k.lower(): v for k, v in results.items()}
        self.log_dict(results, on_step=True, on_epoch=False, prog_bar=True)

    def on_train_epoch_end(self):
        self.aggregate_and_logging(self.train_step_outputs, 'train/loss', step=False)
        self.train_step_outputs.clear()
        self.meter_age_train.reset()
        self.meter_gender_train.reset()

    def on_validation_batch_end(self, outputs: dict[str, Any], batch: Any, batch_idx: int):
        outputs['valid/loss'] = outputs.pop('loss')
        outputs['valid/loss_age'] = outputs.pop('loss_age')
        outputs['valid/loss_gender'] = outputs.pop('loss_gender')
        self.validation_step_outputs.append(outputs)

        # update metrics
        self.meter_age_valid(outputs['pred_age'], batch['age'])
        self.meter_gender_valid(outputs['pred_gender'], F.one_hot(batch['gender']))

    def on_validation_epoch_end(self):
        self.aggregate_and_logging(self.validation_step_outputs, 'valid/loss',  step=False)
        results = self.meter_age_valid.compute()
        results.update(self.meter_gender_valid.compute())
        results = {k.lower(): v for k, v in results.items()}
        self.log_dict(results, on_step=False, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.clear()
        self.meter_age_valid.reset()
        self.meter_gender_valid.reset()

    def _preprocess_gender(self, gender_list):
        return torch.tensor([1 if g == 'm' else 0 if g == 'f' else 255 for g in gender_list]).long()
