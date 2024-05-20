import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class PeriodicCosineAnnealingLR(_LRScheduler):
    """
    One cycle cosine annealing scheduler with warmup.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_epoch (int): The number of epochs.
        iter_per_epoch (int): The number of iterations per epoch.
        warmup_epochs (int): The number of epochs for warmup. Default: 0.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer: Optimizer, max_epoch: int, iter_per_epoch: int,
                 warmup_epochs: int = 0, eta_min: float = 0.0, last_epoch: int = -1):
        assert max_epoch > 0, 'max_epoch should be positive'
        assert iter_per_epoch > 0, 'iter_per_epoch should be positive'
        self.iter_max = max_epoch * iter_per_epoch
        self.iter_warmup = warmup_epochs * iter_per_epoch
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.iter_warmup:
            # Warmup phase: Linear increase
            warmup_factor = self.last_epoch / self.iter_warmup
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            cosine_epochs = self.last_epoch - self.iter_warmup
            cosine_total_epochs = self.iter_max - self.iter_warmup
            cosine_factor = 0.5 * (1 + math.cos(math.pi * cosine_epochs / cosine_total_epochs))
            return [self.eta_min + (base_lr - self.eta_min) * cosine_factor for base_lr in self.base_lrs]
