import os
from argparse import ArgumentParser
from functools import partial

import h5py
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchvision.transforms import transforms

from gaze_estimation.datasets.RTGENEDataset import RTGENEH5Dataset
from gaze_estimation.model import GazeEstimationModelVGG, GazeEstimationModelResNet, ResNet18Dec
from gaze_estimation.training.LAMB import LAMB
from gaze_estimation.training.utils import GazeAngleAccuracyMetric


class TrainRTGENE(pl.LightningModule):

    def __init__(self, hparams, train_subjects, validate_subjects, test_subjects):
        super(TrainRTGENE, self).__init__()
        _loss_fn = {
            "mse": torch.nn.MSELoss,
            "mae": torch.nn.L1Loss,
            "huber": partial(torch.nn.HuberLoss, delta=0.1),  # 0.1 radians is ~6 degrees
            "smooth-l1": partial(torch.nn.SmoothL1Loss, beta=0.1)  # 0.1 radians is ~6 degrees
        }
        _models = {
            "vgg16": partial(GazeEstimationModelVGG, num_out=2),
            "resnet18": partial(GazeEstimationModelResNet, num_out=2)
        }

        self._model = _models.get(hparams.model_base)()
        self._left_reconstruction = ResNet18Dec()
        self._right_reconstruction = ResNet18Dec()
        self._left_reconstruction_loss = torch.nn.MSELoss()
        self._right_reconstruction_loss = torch.nn.MSELoss()
        self._reconstruction_transform = transforms.Compose([transforms.Resize(size=(64, 64)),
                                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                  std=[0.229, 0.224, 0.225])])

        self._criterion = _loss_fn.get(hparams.loss_fn)()
        self._metrics = MetricCollection([GazeAngleAccuracyMetric(reduction="mean")])
        self._train_subjects = train_subjects
        self._validate_subjects = validate_subjects
        self._test_subjects = test_subjects
        self.save_hyperparameters(hparams, ignore=["train_subjects", "validate_subjects", "test_subjects"])
        self._angle_beta = torch.tensor(0.0, dtype=torch.float32)

    def forward(self, left_patch, right_patch, head_pose):
        return self._model(left_patch, right_patch, head_pose)

    def shared_step(self, batch):
        left_patch, right_patch, headpose_label, gaze_labels = batch

        angle_out, bottle_neck = self.forward(left_patch, right_patch, headpose_label)
        metrics = self._metrics(angle_out[:, :2], gaze_labels)

        l_reconstruction = self._left_reconstruction(bottle_neck)
        r_reconstruction = self._left_reconstruction(bottle_neck)

        l_loss = self.hparams.reconstruction_lambda * self._left_reconstruction_loss(l_reconstruction, self._reconstruction_transform(left_patch))
        r_loss = self.hparams.reconstruction_lambda * self._right_reconstruction_loss(r_reconstruction, self._reconstruction_transform(right_patch))

        angle_loss = self._angle_beta * self._criterion(angle_out, gaze_labels)
        loss = angle_loss + l_loss + r_loss

        metrics["loss"] = loss
        metrics["angle_loss"] = angle_loss
        metrics["reconstruction_loss"] = l_loss + r_loss
        metrics["angle_beta"] = self._angle_beta

        return metrics, angle_out, l_reconstruction, r_reconstruction

    def training_step(self, batch, batch_idx):
        result, _, _, _, = self.shared_step(batch)
        train_result = {"train_" + k: v for k, v in result.items()}
        self.log_dict(train_result)
        return result["loss"]

    def validation_step(self, batch, batch_idx):
        result, _, left_reconstruction, right_reconstruction = self.shared_step(batch)
        valid_result = {"valid_" + k: v for k, v in result.items()}
        self.log_dict(valid_result)

        grid = torchvision.utils.make_grid(left_reconstruction[:64])
        self.logger.experiment.add_image('left_generated', grid, self.current_epoch)

        return result["loss"]

    def test_step(self, batch, batch_idx):
        result, _, _, _ = self.shared_step(batch)
        test_result = {"test_" + k: v for k, v in result.items()}
        self.log_dict(test_result)
        return result["loss"]

    def on_train_epoch_end(self) -> None:
        current_beta = -1 * (self.current_epoch - self.hparams.warm_up_50)
        self._angle_beta = 1.0 / (1.0 + torch.exp(torch.tensor(current_beta)))
        self._metrics.reset()  # we're logging a dict, so we need to do this manually

    def configure_optimizers(self):
        params_to_update = []
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)

        optimizer = LAMB(params_to_update, lr=self.hparams.learning_rate)

        return optimizer

    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.RandomResizedCrop(size=(36, 60), scale=(0.5, 1.3)),
                                        transforms.RandomGrayscale(p=0.1),
                                        transforms.ColorJitter(brightness=0.5, hue=0.2, contrast=0.5, saturation=0.5),
                                        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.1),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
        _data = RTGENEH5Dataset(h5_pth=self.hparams.hdf5_file, subject_list=self._train_subjects, transform=transform)
        return DataLoader(_data, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_io_workers, pin_memory=True)

    def val_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(size=(36, 60)),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
        _data = RTGENEH5Dataset(h5_pth=self.hparams.hdf5_file, subject_list=self._validate_subjects, transform=transform)
        return DataLoader(_data, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_io_workers, pin_memory=True)

    def test_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(size=(36, 60)),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
        _data = RTGENEH5Dataset(h5_pth=self.hparams.hdf5_file, subject_list=self._test_subjects, transform=transform)
        return DataLoader(_data, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_io_workers, pin_memory=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--loss_fn', choices=["mse", "mae", "huber", "smooth-l1"], default="mse")
        parser.add_argument('--batch_size', default=128, type=int)
        parser.add_argument('--learning_rate', type=float, default=3e-4)
        parser.add_argument('--reconstruction_lambda', type=float, default=1e-2)  # balances empiric range of reconstruction loss with angular loss
        parser.add_argument('--model_base', choices=["vgg16", "resnet18"], default="resnet18")
        parser.add_argument('--warm_up_50', type=int, default=5, help="Epoch at which warm up phase increases to 50%")
        return parser


if __name__ == "__main__":
    from pytorch_lightning import Trainer
    import psutil

    root_dir = os.path.dirname(os.path.realpath(__file__))

    root_parser = ArgumentParser(add_help=False)
    root_parser.add_argument('--gpu', type=int, default=1,
                             help='gpu to use, can be repeated for mutiple gpus i.e. --gpu 1 --gpu 2', action="append")
    root_parser.add_argument('--hdf5_file', type=str, default="rtgene_dataset.hdf5")
    root_parser.add_argument('--dataset_type', type=str, choices=["rt_gene", "other"], default="rt_gene")
    root_parser.add_argument('--num_io_workers', default=psutil.cpu_count(logical=False), type=int)
    root_parser.add_argument('--k_fold_validation', action="store_true", dest="k_fold_validation")
    root_parser.add_argument('--all_dataset', action='store_false', dest="k_fold_validation")
    root_parser.add_argument('--seed', type=int, default=0)
    root_parser.add_argument('--min_epochs', type=int, default=5, help="Number of Epochs to perform at a minimum")
    root_parser.add_argument('--max_epochs', type=int, default=300, help="Maximum number of epochs to perform; the trainer will Exit after.")
    root_parser.set_defaults(k_fold_validation=True)

    model_parser = TrainRTGENE.add_model_specific_args(root_parser)
    hyperparams = model_parser.parse_args()
    hyperparams.hdf5_file = os.path.abspath(os.path.expanduser(hyperparams.hdf5_file))

    pl.seed_everything(hyperparams.seed)

    train_subs = []
    valid_subs = []
    test_subs = []
    if hyperparams.dataset_type == "rt_gene":
        if hyperparams.k_fold_validation:
            train_subs.append([1, 2, 8, 10, 3, 4, 7, 9])
            train_subs.append([1, 2, 8, 10, 5, 6, 11, 12, 13])
            train_subs.append([3, 4, 7, 9, 5, 6, 11, 12, 13])
            # validation set is always subjects 14, 15 and 16
            valid_subs.append([0, 14, 15, 16])
            valid_subs.append([0, 14, 15, 16])
            valid_subs.append([0, 14, 15, 16])
            # test subjects
            test_subs.append([5, 6, 11, 12, 13])
            test_subs.append([3, 4, 7, 9])
            test_subs.append([1, 2, 8, 10])
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
        model = TrainRTGENE(hparams=hyperparams, train_subjects=train_s, validate_subjects=valid_s, test_subjects=test_s)
        # save all models
        checkpoint_callback = ModelCheckpoint(monitor='valid_GazeAngleAccuracyMetric', mode='min', verbose=False, save_top_k=10)

        # start training
        trainer = Trainer(gpus=hyperparams.gpu,
                          precision=32,
                          callbacks=[checkpoint_callback],
                          min_epochs=hyperparams.min_epochs,
                          max_epochs=hyperparams.max_epochs)
        trainer.fit(model)
        # trainer.test()
