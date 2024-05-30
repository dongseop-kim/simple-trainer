from torchmetrics.classification import MultilabelConfusionMatrix

import torch


class ConfusionMatrix(MultilabelConfusionMatrix):
    '''
    Multi-label confusion matrix metric.

    C00: True negatives
    C01: False positives
    C10: False negatives
    C11: True positives
    '''

    def __init__(self, num_labels: int, prefix: str = ''):
        super().__init__(num_labels)
        self.prefix = prefix

    def compute(self) -> dict[str, torch.Tensor]:
        confusion_matrix: torch.Tensor = super().compute()
        assert len(confusion_matrix.shape) == 3

        # True negatives, False positives, False negatives, True positives
        tps = confusion_matrix[:, 1, 1]  # N
        fps = confusion_matrix[:, 0, 1]  # N
        fns = confusion_matrix[:, 1, 0]  # N
        tns = confusion_matrix[:, 0, 0]  # N

        accuracys = (tps + tns) / (tps + fps + fns + tns + 1e-8)
        precisions = tps / (tps + fps + 1e-8)
        sensitivity = tps / (tps + fns + 1e-8)
        specificity = tns / (tns + fps + 1e-8)
        f1_scores = 2 * tps / (2 * tps + fps + fns + 1e-8)

        scores = {'acc': torch.mean(accuracys), 'prec': torch.mean(precisions), 'sens': torch.mean(sensitivity),
                  'spec': torch.mean(specificity), 'f1': torch.mean(f1_scores)}

        for i, (acc, prec, sens, spec, f1) in enumerate(zip(accuracys, precisions, sensitivity, specificity, f1_scores)):
            scores.update({f'acc_{i}': acc, f'prec_{i}': prec, f'sens_{i}': sens, f'spec_{i}': spec, f'f1_{i}': f1})

        # add prefix
        if self.prefix:
            scores = {f'{self.prefix}/{k}': v for k, v in scores.items()}
        return scores
