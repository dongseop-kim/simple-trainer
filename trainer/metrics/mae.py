import torch
from torch import Tensor as T
from torchmetrics import MeanAbsoluteError


class MAE(MeanAbsoluteError):
    def update(self, preds: T, target: T):
        preds: T = torch.sum(preds.flatten(start_dim=1), dim=1)
        assert preds.shape == target.shape, f"preds shape {preds.shape} != target shape {target.shape}"
        return super().update(preds.detach(), target.detach())
