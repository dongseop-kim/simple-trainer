import torch
from pytorch_lightning import LightningModule

from trainer.models import Model


class BaseEngine(LightningModule):
    def __init__(self, model: Model, optimizer=None, scheduler=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        # each step outputs
        self.train_step_outputs: list[dict[str, torch.Tensor]] = []
        self.validation_step_outputs: list[dict[str, torch.Tensor]] = []
        self.test_step_outputs: list[dict[str, torch.Tensor]] = []
        self.predict_step_outputs: list[dict[str, torch.Tensor]] = []

    def configure_optimizers(self):
        return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler}

    def aggregate_and_logging(self, outputs: list[dict[str, torch.Tensor]], key: str,
                              prefix: str = None, is_step: bool = False):
        '''Aggregate outputs from each step and log'''
        loss_per_epoch = torch.stack([x[key] for x in outputs]).mean()
        prefix = '' if prefix is None else prefix + '/'
        self.log(f'{prefix}{key}', loss_per_epoch, on_step=is_step, on_epoch=not is_step, prog_bar=True)
