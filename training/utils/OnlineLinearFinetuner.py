from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from pytorch_lightning import Callback
from torchmetrics.regression.mse import MeanSquaredError
from torchmetrics import MetricCollection
from gaze_estimation.training.utils.GazeAngleAccuracyMetric import GazeAngleAccuracyMetric
from typing import Sequence, Union, Tuple, Optional


class OnlineFineTuner(Callback):

    def __init__(self, encoder_output_dim: int = 128) -> None:
        super().__init__()

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.encoder_output_dim = encoder_output_dim
        self.metrics = MetricCollection([MeanSquaredError(), GazeAngleAccuracyMetric()])
        self.loss_fn = torch.nn.SmoothL1Loss(beta=0.1)  # roughly 6 degrees

    def on_pretrain_routine_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # add linear_eval layer and optimizer
        pl_module.online_finetuner = MLPEvaluator(n_input=self.encoder_output_dim).to(pl_module.device)
        self.optimizer = torch.optim.Adam(pl_module.online_finetuner.parameters(), lr=1e-4, weight_decay=1e-5)
        self.metrics.to(pl_module.device)

    @staticmethod
    def to_device(batch: Sequence, device: Union[str, torch.device]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eye_patch, _, head_pose, gaze_pose = batch
        eye_patch = eye_patch.to(device)
        head_pose = head_pose.to(device)
        gaze_pose = gaze_pose.to(device)

        return eye_patch, head_pose, gaze_pose

    def shared_batch_end(self, pl_module: "pl.LightningModule", batch):
        x, head_pose, gaze_pose = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            with set_training(pl_module, False):
                feats, _, _ = pl_module(x)

        preds = pl_module.online_finetuner(feats, head_pose)
        loss = self.loss_fn(preds, gaze_pose)

        metric_results = self.metrics(preds, gaze_pose)
        metric_results["loss"] = loss
        return metric_results

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int,
                           unused: Optional[int] = 0) -> None:
        metric_results = self.shared_batch_end(pl_module=pl_module, batch=batch)

        metric_results["loss"].backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        metric_results = {"online_train_" + k: v for k, v in metric_results.items()}
        pl_module.log_dict(metric_results, on_step=True, on_epoch=False)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.metrics.reset()

    def on_validation_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Sequence,
            batch: Sequence,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        metric_results = self.shared_batch_end(pl_module=pl_module, batch=batch)
        metric_results = {"online_valid_" + k: v for k, v in metric_results.items()}
        pl_module.log_dict(metric_results, on_step=True, on_epoch=False, sync_dist=True)


@contextmanager
def set_training(module: nn.Module, mode: bool):
    original_mode = module.training

    try:
        module.train(mode)
        yield module
    finally:
        module.train(original_mode)


class MLPEvaluator(nn.Module):
    def __init__(self, n_input: int = 128, n_classes=2, n_hidden=512, p=0.05):
        super().__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden

        # use simple MLP classifier
        self.block_forward = nn.Sequential(
            Flatten(),
            nn.Dropout(p=p),
            nn.Linear(n_input + 2, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.GELU(),
            nn.Dropout(p=p),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.GELU(),
            nn.Dropout(p=p),
            nn.Linear(n_hidden, n_classes)
        )

    def forward(self, latent, headpose):
        x = torch.concat([latent, headpose], dim=1)
        logits = self.block_forward(x)
        return logits


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)
