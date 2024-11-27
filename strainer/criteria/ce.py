from torch import Tensor as T
from torch import nn


class CrossEntropy(nn.CrossEntropyLoss):
    def __init__(self, ignore_index: int = 255, reduction: str = 'mean', label_smoothing: float = 0.0):
        super().__init__()
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction,
                                            label_smoothing=label_smoothing)

    def forward(self, logit: T, target: T) -> dict[str, T]:
        logit = logit.float()
        target = target.long()
        loss: T = self.criteria(logit, target)
        return {'loss': loss}


class BinaryCrossEntropy(nn.BCEWithLogitsLoss):
    def __init__(self, reduction: str = 'mean', label_smoothing: float = 0.0):
        super().__init__()
        self.criteria = nn.BCEWithLogitsLoss(reduction=reduction)
        self.label_smoothing = label_smoothing

    def forward(self, logit: T, target: T) -> dict[str, T]:
        if target.shape != logit.shape:
            target = target.view(logit.shape)
        logit = logit.float()
        target = target.float()

        # Label smoothing when label_smoothing > 0
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + self.label_smoothing / 2

        loss: T = self.criteria(logit, target)

        return {'loss': loss}
