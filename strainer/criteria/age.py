import torch
from torch import Tensor as T
from torch import nn


class CoralLoss(nn.Module):
    """
    Coral Loss in "Rank consistent ordinal regression for neural networks with application to age estimation."

    Args:
        num_classes (int): Number of ordinal classes.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.nc = num_classes
        self.critrion = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, logit: T, target: T) -> dict[str, T]:
        target = target.long()
        if len(target.shape) > 1:
            target = target.squeeze()
            assert len(target.shape) == 1, f"target shape must be (N, ) but {target.shape}"

        N = target.shape[0]
        logit = logit.view(N, self.nc)
        target = self._label_to_rank(target)
        loss = self.critrion(logit, target)
        return {'loss': loss}

    def _label_to_rank(self, target: T) -> T:
        """
        Convert class labels to ordinal rankings.

        Args:
            target (torch.Tensor): Class labels.

        Returns:
            torch.Tensor: Ordinal rankings.
        """
        ranks = torch.arange(self.nc, device=target.device, dtype=torch.float32)  # self.nc
        ranks = ranks.unsqueeze(0)  # 1 x self.nc
        new_target = (ranks < target.unsqueeze(1)).float()  # N x self.nc
        # target == torch.sum(new_target, dim=1) = True 임.
        return new_target
