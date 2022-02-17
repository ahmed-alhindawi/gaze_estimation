import itertools
import math
import os
from argparse import ArgumentParser
from functools import partial

import h5py
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchvision.transforms import transforms

from gaze_estimation.datasets.RTGENEDataset import RTGENEH5Dataset
from gaze_estimation.model import GazeEncoder, ResNet18Dec
from gaze_estimation.training.LAMB import LAMB
from gaze_estimation.training.utils.OnlineLinearFinetuner import OnlineFineTuner


class TrainRTGENEVAE(pl.LightningModule):

    def __init__(self, hparams, train_subjects, validate_subjects, test_subjects):
        super(TrainRTGENEVAE, self).__init__()

        self.encoder = GazeEncoder(latent_dim=hparams.latent_dim)
        self.decoder = ResNet18Dec(z_dim=hparams.latent_dim)
        self._train_subjects = train_subjects
        self._validate_subjects = validate_subjects
        self._test_subjects = test_subjects
        self.save_hyperparameters(hparams, ignore=["train_subjects", "validate_subjects", "test_subjects"])

    def forward(self, img):
        mu, logvar = self.encoder(img)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z, mu, logvar

    def shared_step(self, batch):
        img, _, _, _ = batch

        z, mu, logvar = self.forward(img)
        reconstruction = self.decoder(z)

        recons_loss = F.mse_loss(img, reconstruction)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

        loss = recons_loss + self.hparams.kld_weight * kld_loss
        result = {"kld_loss": kld_loss,
                  "mse_loss": recons_loss,
                  "loss": loss}
        return result, reconstruction

    def training_step(self, batch, batch_idx):
        result, _ = self.shared_step(batch)
        train_result = {"train_" + k: v for k, v in result.items()}
        self.log_dict(train_result)
        return result["loss"]

    def validation_step(self, batch, batch_idx):
        result, recon = self.shared_step(batch)
        valid_result = {"valid_" + k: v for k, v in result.items()}
        self.log_dict(valid_result)

        grid = torchvision.utils.make_grid(recon[:64])
        self.logger.experiment.add_image('reconstruction', grid, self.current_epoch)

        return result["loss"]

    def test_step(self, batch, batch_idx):
        result, _ = self.shared_step(batch)
        test_result = {"test_" + k: v for k, v in result.items()}
        self.log_dict(test_result)
        return result["loss"]

    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.3)),
                                        transforms.RandomGrayscale(p=0.1),
                                        transforms.ColorJitter(brightness=0.5, hue=0.2, contrast=0.5, saturation=0.5),
                                        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.1),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
        _data = RTGENEH5Dataset(h5_pth=self.hparams.hdf5_file, subject_list=self._train_subjects, transform=transform)
        return DataLoader(_data, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_io_workers, pin_memory=True)

    def val_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(size=(64, 64)),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
        _data = RTGENEH5Dataset(h5_pth=self.hparams.hdf5_file, subject_list=self._validate_subjects, transform=transform)
        return DataLoader(_data, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_io_workers, pin_memory=True)

    def test_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(size=(64, 64)),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
        _data = RTGENEH5Dataset(h5_pth=self.hparams.hdf5_file, subject_list=self._test_subjects, transform=transform)
        return DataLoader(_data, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_io_workers, pin_memory=True)

    @staticmethod
    def linear_warmup_decay(warmup_steps, total_steps, cosine=True):
        """
        Linear warmup for warmup_steps, optionally with cosine annealing or
        linear decay to 0 at total_steps

        Adapted from grid_ai/aavae
        """

        def fn(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))

            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            if cosine:
                # cosine decay
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            # linear decay
            return 1.0 - progress

        return fn

    def configure_optimizers(self):

        params_to_update = []
        for name, param in itertools.chain(self.encoder.named_parameters(), self.decoder.named_parameters()):
            if param.requires_grad:
                params_to_update.append(param)

        if self.hparams.optimiser == "adam_w":
            optimizer = torch.optim.AdamW(params_to_update, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimiser == "lamb":
            optimizer = LAMB(params_to_update, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        else:
            raise NotImplementedError("Only 'adam_w' and 'lamb' optimisers have been implemented")

        if self.hparams.optimiser_schedule:

            warmup_steps = (81152 / self.hparams.batch_size) * self.hparams.warmup_epochs
            total_steps = (81152 / self.hparams.batch_size) * self.hparams.max_epochs

            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, linear_warmup_decay(warmup_steps, total_steps, self.hparams.cosine_decay)),
                "interval": "step",
                "frequency": 1,
            }

            return [optimizer], [scheduler]
        else:
            return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--batch_size', default=128, type=int)
        parser.add_argument('--learning_rate', type=float, default=3e-2)
        parser.add_argument('--weight_decay', default=1e-5, type=float)
        parser.add_argument('--optimiser_schedule', type=bool, default=False)
        parser.add_argument('--warmup_epochs', type=int, default=5)
        parser.add_argument('--cosine_decay', action="store_true", dest="cosine_decay")
        parser.add_argument('--linear_decay', action="store_false", dest="cosine_decay")
        parser.add_argument('--decay_reconstruction_loss', type=bool, default=True)
        parser.add_argument('--optimiser', choices=["adam_w", "lamb"], default="adam_w")
        parser.add_argument('--kld_weight', type=float, default=0.00025)
        parser.add_argument('--latent_dim', type=int, default=128)
        parser.set_defaults(cosine_decay=True)
        return parser


if __name__ == "__main__":
    from pytorch_lightning import Trainer
    from gaze_estimation.training.utils import print_args
    import psutil

    root_dir = os.path.dirname(os.path.realpath(__file__))

    root_parser = ArgumentParser(add_help=False)
    root_parser.add_argument('--gpu', type=int, default=-1, help="number of gpus to use")
    root_parser.add_argument('--hdf5_file', type=str, default="rtgene_dataset.hdf5")
    root_parser.add_argument('--dataset_type', type=str, choices=["rt_gene", "other"], default="rt_gene")
    root_parser.add_argument('--num_io_workers', default=psutil.cpu_count(logical=False), type=int)
    root_parser.add_argument('--k_fold_validation', action="store_true", dest="k_fold_validation")
    root_parser.add_argument('--all_dataset', action='store_false', dest="k_fold_validation")
    root_parser.add_argument('--seed', type=int, default=0)
    root_parser.add_argument('--min_epochs', type=int, default=5, help="Number of Epochs to perform at a minimum")
    root_parser.add_argument('--max_epochs', type=int, default=300, help="Maximum number of epochs to perform; the trainer will Exit after.")
    root_parser.set_defaults(k_fold_validation=True)

    model_parser = TrainRTGENEVAE.add_model_specific_args(root_parser)
    hyperparams = model_parser.parse_args()
    hyperparams.hdf5_file = os.path.abspath(os.path.expanduser(hyperparams.hdf5_file))
    print_args(hyperparams)

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
        model = TrainRTGENEVAE(hparams=hyperparams, train_subjects=train_s, validate_subjects=valid_s, test_subjects=test_s)

        # can't save valid_loss or valid_angle_loss as now that's a composite loss mediated by a training related parameter
        callbacks = [ModelCheckpoint(monitor='valid_loss', mode='min', verbose=False, save_top_k=10,
                                     filename="epoch={epoch}-valid_loss={valid_loss:.2f}", save_last=True),
                     LearningRateMonitor(),
                     OnlineFineTuner(encoder_output_dim=hyperparams.latent_dim)]

        # start training
        trainer = Trainer(gpus=hyperparams.gpu,
                          precision=32,
                          callbacks=callbacks,
                          min_epochs=hyperparams.min_epochs,
                          log_every_n_steps=10,
                          max_epochs=hyperparams.max_epochs)
        trainer.fit(model)
        # trainer.test()