from typing import Any, Callable, List, Optional
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat


class GazeAngleAccuracyMetric(Metric):
    is_differentiable = True
    higher_is_better = False
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
            self,
            reduction: str = "mean",
            compute_on_step: bool = True,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
            dist_sync_fn: Callable = None,
    ) -> None:
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step, process_group=process_group, dist_sync_fn=dist_sync_fn)

        _reduction_mappings = {
            "sum": torch.sum,
            "mean": torch.mean,
            "none": lambda x: x,
            None: lambda x: x,
        }
        if reduction not in _reduction_mappings.keys():
            raise ValueError(f"Expected argument `reduction` to be one of {_reduction_mappings.keys()} but got {reduction}")

        self._reduction_strategy = _reduction_mappings[reduction]

        self.add_state("preds", [], dist_reduce_fx="cat")
        self.add_state("target", [], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.
                Args:
                    preds: Predicted tensor with shape ``(N,2)``
                    target: Ground truth tensor with shape ``(N,2)``
                """
        if preds.shape != target.shape:
            raise RuntimeError("Predictions and targets are expected to have the same shape")
        preds = preds.float()
        target = target.float()

        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)

        pred_x = -1 * torch.cos(preds[:, 0]) * torch.sin(preds[:, 1])
        pred_y = -1 * torch.sin(preds[:, 0])
        pred_z = -1 * torch.cos(preds[:, 0]) * torch.cos(preds[:, 1])
        pred = torch.vstack([pred_x, pred_y, pred_z]).T
        pred_norm = (pred.T / torch.norm(pred, dim=1)).T

        true_x = -1 * torch.cos(target[:, 0]) * torch.sin(target[:, 1])
        true_y = -1 * torch.sin(target[:, 0])
        true_z = -1 * torch.cos(target[:, 0]) * torch.cos(target[:, 1])
        gt = torch.vstack([true_x, true_y, true_z]).T
        gt_norm = (gt.T / torch.norm(gt, dim=1)).T

        acc = torch.rad2deg(torch.arccos(torch.sum(pred_norm * gt_norm, dim=1)))

        return self._reduction_strategy(acc)
