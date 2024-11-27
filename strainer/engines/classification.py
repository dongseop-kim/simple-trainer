from typing import Any

from torch import Tensor as T

from strainer.engines.base import BaseEngine
from strainer.metrics.confusion_matrix import ConfusionMatrix
from strainer.models import ModelBase


class MultiLabelClassifier(BaseEngine):
    def __init__(self, model: ModelBase, optimizer=None, scheduler=None, criterion=None, num_labels: int = 2):
        super().__init__(model, optimizer, scheduler, criterion)

        self.meter_train = ConfusionMatrix(num_labels=num_labels, prefix='train')
        self.meter_valid = ConfusionMatrix(num_labels=num_labels, prefix='valid')

    def on_train_batch_end(self, outputs: dict[str, Any], batch: Any, batch_idx: int):
        outputs['train/loss'] = outputs.pop('loss')
        self.train_step_outputs.append(outputs)

        # update metrics
        self.meter_train(outputs['pred'].squeeze(), batch['label'])

        # logging
        results: dict[str, T] = self.meter_train.compute()
        results = {k.lower(): v for k, v in results.items()}
        self.log_dict(results, on_step=True, on_epoch=False, prog_bar=True)

    def on_train_epoch_end(self):
        self.aggregate_and_logging(self.train_step_outputs, 'train/loss', step=False)
        self.train_step_outputs.clear()
        self.meter_train.reset()

    def on_validation_batch_end(self, outputs: dict[str, Any], batch: Any, batch_idx: int):
        outputs['valid/loss'] = outputs.pop('loss')
        self.validation_step_outputs.append(outputs)

        # update metrics
        self.meter_valid(outputs['pred'].squeeze(), batch['label'])

    def on_validation_epoch_end(self):
        self.aggregate_and_logging(self.validation_step_outputs, 'valid/loss', step=False)
        results = self.meter_valid.compute()
        results = {k.lower(): v for k, v in results.items()}
        self.log_dict(results, on_step=False, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.clear()
        self.meter_valid.reset()
