import torch
from torchmetrics import MeanAbsoluteError


class MAE(MeanAbsoluteError):
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.sum(preds.flatten(start_dim=1), dim=1)
        assert preds.shape == target.shape, f"preds shape {preds.shape} != target shape {target.shape}"
        return super().update(preds, target)
