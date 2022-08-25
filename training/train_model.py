import math
import os
from argparse import ArgumentParser
from functools import partial

import h5py
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchvision.transforms import transforms

from gaze_estimation.datasets.RTGENEDataset import RTGENEFileDataset, RTGENEH5Dataset
from gaze_estimation.model import GazeEstimationModelVGG, GazeEstimationModelResNet
from gaze_estimation.training.LAMB import LAMB
from gaze_estimation.training.utils import GazeAngleAccuracyMetric


class GNLL(torch.nn.Module):
    def __init__(self):
        super(GNLL, self).__init__()

    def forward(self, angle_out, label, var):
        loss = 0.5 * ((torch.exp(-var) * (angle_out - label) ** 2) + var)
        return loss.mean()


LOSS_FN = {
    "mse": {"loss": torch.nn.MSELoss, "num_out": 2},
    "mae": {"loss": torch.nn.L1Loss, "num_out": 2},
    "smooth-l1": {"loss": partial(torch.nn.SmoothL1Loss, beta=0.1), "num_out": 2},  # 0.1 radians is ~6 degrees
    "gnll": {"loss": GNLL, "num_out": 3}
}
MODELS = {
    "vgg16": GazeEstimationModelVGG,
    "resnet18": GazeEstimationModelResNet
}

OPTIMISERS = {
    "adam_w": torch.optim.AdamW,
    "lamb": LAMB
}


class TrainRTGENE(pl.LightningModule):

    def __init__(self, hparams, train_subjects, validate_subjects, test_subjects, dataset):
        super(TrainRTGENE, self).__init__()

        self.model = MODELS[hparams.model_base](num_out=LOSS_FN[hparams.loss_fn]["num_out"])
        self._dataset = dataset
        self._criterion = LOSS_FN[hparams.loss_fn]["loss"]()
        self._metrics = MetricCollection([GazeAngleAccuracyMetric(reduction="mean")])
        self._train_subjects = train_subjects
        self._validate_subjects = validate_subjects
        self._test_subjects = test_subjects
        self.save_hyperparameters(hparams)

    def forward(self, left_patch, right_patch, head_pose):
        return self.model(left_patch, right_patch, head_pose)

    def shared_step(self, batch):
        _, _, left_patch, right_patch, headpose_label, gaze_labels = batch

        y, _ = self.forward(left_patch, right_patch, headpose_label)
        angle_out = y[:, :2]
        metrics = self._metrics(angle_out[:, :2], gaze_labels)

        if self.hparams.loss_fn == "gnll":
            var_out = y[:, 2].unsqueeze(-1).exp()  # homoscedastic
            loss = self._criterion(angle_out, gaze_labels, var_out)
        else:
            loss = self._criterion(angle_out, gaze_labels)

        metrics["loss"] = loss

        return metrics, angle_out

    def training_step(self, batch, batch_idx):
        result, _ = self.shared_step(batch)
        train_result = {"train_" + k: v for k, v in result.items()}
        self.log_dict(train_result)
        return result["loss"]

    def validation_step(self, batch, batch_idx):
        result, _ = self.shared_step(batch)
        valid_result = {"valid_" + k: v for k, v in result.items()}
        self.log_dict(valid_result)

        return result["loss"]

    def test_step(self, batch, batch_idx):
        result, _ = self.shared_step(batch)
        test_result = {"test_" + k: v for k, v in result.items()}
        self.log_dict(test_result)
        return result["loss"]

    def on_train_epoch_end(self) -> None:
        self._metrics.reset()  # we're logging a dict, so we need to do this manually

    def train_dataloader(self):
        eye_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.RandomResizedCrop(size=(36, 60), scale=(0.5, 1.3)),
                                            transforms.RandomGrayscale(p=0.1),
                                            transforms.ColorJitter(brightness=0.5, hue=0.2, contrast=0.5, saturation=0.5),
                                            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.1),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])

        data = self._dataset(subject_list=self._train_subjects, eye_transform=eye_transform)
        return DataLoader(data, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_io_workers, pin_memory=True)

    def val_dataloader(self):
        data = self._dataset(subject_list=self._validate_subjects, eye_transform=None)
        return DataLoader(data, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_io_workers, pin_memory=True)

    def test_dataloader(self):
        data = self._dataset(subject_list=self._test_subjects, eye_transform=None)
        return DataLoader(data, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_io_workers, pin_memory=True)

    def configure_optimizers(self):
        # def linear_warmup_decay(warmup_steps):
        #     def fn(step):
        #         if step < warmup_steps:
        #             return float(step) / float(max(1, warmup_steps))
        #
        #         return 1.0
        #
        #     return fn

        def linear_warmup_decay(warmup_steps, total_steps, cosine=True, linear=False):
            """
            Linear warmup for warmup_steps, optionally with cosine annealing or
            linear decay to 0 at total_steps
            """
            # check if both decays are not True at the same time
            assert not (linear and cosine)

            def fn(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))

                if not (cosine or linear):
                    # no decay
                    return 1.0

                progress = float(step - warmup_steps) / float(
                    max(1, total_steps - warmup_steps)
                )
                if cosine:
                    # cosine decay
                    return 0.5 * (1.0 + math.cos(math.pi * progress))

                # linear decay
                return 1.0 - progress

            return fn

        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)

        optimizer = OPTIMISERS.get(self.hparams.optimiser)(params_to_update, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

        if self.hparams.optimiser_schedule:
            warmup_steps = (74804 / self.hparams.batch_size) * self.hparams.warmup_epochs  # 74804 is from len(training_dataset)
            total_steps = (74804 / self.hparams.batch_size) * self.hparams.max_epochs

            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, linear_warmup_decay(warmup_steps, total_steps=total_steps)),
                "interval": "step",
                "frequency": 1,
            }

            return [optimizer], [scheduler]
        else:
            return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--loss_fn', choices=LOSS_FN.keys(), default=list(LOSS_FN.keys())[0])
        parser.add_argument('--batch_size', default=128, type=int)
        parser.add_argument('--learning_rate', type=float, default=3e-2)
        parser.add_argument('--model_base', choices=MODELS.keys(), default=list(MODELS.keys())[0])
        parser.add_argument('--weight_decay', default=1e-3, type=float)
        parser.add_argument('--optimiser_schedule', action="store_true", default=False)
        parser.add_argument('--warmup_epochs', type=int, default=10)
        parser.add_argument('--optimiser', choices=OPTIMISERS.keys(), default=list(OPTIMISERS.keys())[0])
        return parser


if __name__ == "__main__":
    from pytorch_lightning import Trainer
    from gaze_estimation.training.utils import print_args
    import psutil

    root_dir = os.path.dirname(os.path.realpath(__file__))

    parser = ArgumentParser(add_help=False)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--dataset_type', nargs='?', choices=('hdf5', 'folder'))
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, choices=["rt_gene", "other"], default="rt_gene")
    parser.add_argument('--num_io_workers', default=psutil.cpu_count(logical=False), type=int)
    parser.add_argument('--k_fold_validation', action="store_true", dest="k_fold_validation")
    parser.add_argument('--all_dataset', action='store_false', dest="k_fold_validation")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32])
    parser.add_argument('--distributed_strategy', choices=["none", "ddp_find_unused_parameters_false"],
                        default="ddp_find_unused_parameters_false")
    parser.add_argument('--max_epochs', type=int, default=300, help="Maximum number of epochs to perform; the trainer will Exit after.")
    parser.add_argument('--stocastic_weighted_averaging', action="store_true", dest="stochastic_weight_averaging", default="False")
    parser.add_argument('--swa_epochs', type=int, default=10)
    parser.add_argument('--swa_lr', type=float, default=1e-4)
    parser.add_argument('--swa_epoch_start', type=float, default=0.8)

    parser.set_defaults(k_fold_validation=True)

    model_parser = TrainRTGENE.add_model_specific_args(parser)
    hyperparams = model_parser.parse_args()
    hyperparams.dataset_path = os.path.abspath(os.path.expanduser(hyperparams.dataset_path))
    print_args(hyperparams)

    pl.seed_everything(hyperparams.seed)

    train_subs = []
    valid_subs = []
    test_subs = []
    if hyperparams.dataset_name == "rt_gene":
        if hyperparams.k_fold_validation:
            train_subs.append([3, 4, 7, 9, 5, 6, 11, 12, 13, 14, 15, 16])
            train_subs.append([1, 2, 8, 10, 5, 6, 11, 12, 13, 14, 15, 16])
            train_subs.append([1, 2, 8, 10, 3, 4, 7, 9, 14, 15, 16])

            valid_subs.append([1, 2, 8, 10])
            valid_subs.append([3, 4, 7, 9])
            valid_subs.append([5, 6, 11, 12, 13])

            test_subs.append([0])
            test_subs.append([0])
            test_subs.append([0])
        else:
            train_subs.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
            valid_subs.append([0])  # Note that this is a hack and should not be used to get results for papers
            test_subs.append([0])
    else:
        file = h5py.File(hyperparams.hdf5_file, mode="r")
        keys = [int(subject[1:]) for subject in list(file.keys())]
        file.close()
        if hyperparams.k_fold_validation:
            all_subjects = range(len(keys))
            for leave_one_out_idx in all_subjects:
                train_subs.append(all_subjects[:leave_one_out_idx] + all_subjects[leave_one_out_idx + 1:])
                valid_subs.append([leave_one_out_idx])  # Note that this is a hack and should not be used to get results for papers
                test_subs.append([leave_one_out_idx])
        else:
            train_subs.append(keys[1:])
            valid_subs.append([keys[0]])
            test_subs.append([keys[0]])

    for fold, (train_s, valid_s, test_s) in enumerate(zip(train_subs, valid_subs, test_subs)):
        # create the dataset for this type
        if hyperparams.dataset_type == "hdf5":
            dataset_partial = partial(RTGENEH5Dataset, root_path=hyperparams.dataset_path)
        elif hyperparams.dataset_type == "folder":
            dataset_partial = partial(RTGENEFileDataset, root_path=hyperparams.dataset_path)
        else:
            raise ValueError(f"Dataset type {hyperparams.dataset_path} not suitable")

        model = TrainRTGENE(hparams=hyperparams, train_subjects=train_s, validate_subjects=valid_s, test_subjects=test_s, dataset=dataset_partial)

        # can't save valid_loss or valid_angle_loss as now that's a composite loss mediated by a training related parameter
        callbacks = [ModelCheckpoint(monitor='valid_loss', mode='min', verbose=False, save_top_k=10, save_last=True,
                                     filename="{epoch}-{valid_loss:.2f}-{valid_GazeAngleAccuracyMetric:.2f}"), LearningRateMonitor()]

        if hyperparams.stochastic_weight_averaging:
            swa = StochasticWeightAveraging(swa_epoch_start=0.8, swa_lrs=hyperparams.swa_lr, annealing_epochs=hyperparams.swa_epochs)
            callbacks.append(swa)

        # start training
        trainer = Trainer(gpus=hyperparams.gpu,
                          precision=int(hyperparams.precision),
                          callbacks=callbacks,
                          strategy=None if hyperparams.distributed_strategy == "none" else hyperparams.distributed_strategy,
                          log_every_n_steps=10,
                          max_epochs=hyperparams.max_epochs)
        trainer.fit(model)
        trainer.test(ckpt_path="best")
