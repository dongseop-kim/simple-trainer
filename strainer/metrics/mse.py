import torch
from torch import Tensor as T
from torchmetrics import MeanSquaredError


class MSE(MeanSquaredError):
    def update(self, preds: T, target: T):
        preds: T = torch.sum(preds.flatten(start_dim=1), dim=1)
        assert preds.shape == target.shape, f"preds shape {preds.shape} != target shape {target.shape}"
        return super().update(preds.detach(), target.detach().float())


class RMSE(MSE):
    def __init__(self, **kwargs):
        super().__init__(squared=False, **kwargs)
