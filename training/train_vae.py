import itertools
import os
from argparse import ArgumentParser
from functools import partial

import h5py
import pytorch_lightning as pl
import torch
import torchvision.utils
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import transforms

from gaze_estimation.datasets.RTGENEDataset import RTGENEH5Dataset
from gaze_estimation.model import resnet18, resnet34, resnet50, decoder18, decoder34, decoder50, ProjectionHeadVAERegression, LatentRegressor
from gaze_estimation.training.LAMB import LAMB
from gaze_estimation.training.utils.OnlineLinearFinetuner import OnlineFineTuner

OPTIMISERS = {
    "adam_w": torch.optim.AdamW,
    "adam": torch.optim.Adam,
    "lamb": LAMB,
    "sgd": torch.optim.SGD
}

ENCODERS = {
    "resnet18": (resnet18, 512),
    "resnet34": (resnet34, 512),
    "resnet50": (resnet50, 1024)
}

DECODERS = {
    "resnet18": decoder18,
    "resnet34": decoder34,
    "resnet50": decoder50
}


class TrainRTGENEVAE(pl.LightningModule):

    def __init__(self, hparams, train_subjects, validate_subjects, test_subjects):
        super(TrainRTGENEVAE, self).__init__()

        extract_img_fn = {
            "left": lambda x: (x[2], x[5]),
            "right": lambda x: (x[3], x[5]),
        }

        self.encoder = ENCODERS[hparams.encoder][0]()
        self.projection = ProjectionHeadVAERegression(input_dim=ENCODERS[hparams.encoder][1], latent_dim=hparams.latent_dim, regressor_dim=2)
        self.latent_regressor = LatentRegressor(input_dim=2, output_dim=hparams.latent_dim)
        self.decoder = decoder18(latent_dim=hparams.latent_dim)
        self._normaliser = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        self._train_subjects = train_subjects
        self._validate_subjects = validate_subjects
        self._test_subjects = test_subjects
        self._extract_fn = extract_img_fn[hparams.eye_laterality]
        self.save_hyperparameters(hparams, ignore=["train_subjects", "validate_subjects", "test_subjects"])

    def forward(self, img):
        result = self.encoder(img)
        return result

    @staticmethod
    def gaussian_likelihood(mean, sample):
        dist = torch.distributions.Normal(mean, 1.0)
        log_pxz = dist.log_prob(sample)

        # sum over dimensions
        return log_pxz.sum(dim=(1, 2, 3))

    @staticmethod
    def sample(z_mu, z_var, eps=1e-6):
        std = torch.exp(0.5 * z_var)

        q = torch.distributions.Normal(z_mu, std)
        z = q.rsample()

        return z

    def shared_step(self, batch):
        img, gaze = self._extract_fn(batch)
        encoding = self.forward(img)
        mu, logvar, mu_r, logvar_r = self.projection(encoding)

        z = self.sample(mu, logvar)
        r = self.sample(mu_r, logvar_r)

        reconstruction = self.decoder(z)
        recons_loss = self.hparams.recon_weight * self.gaussian_likelihood(img, reconstruction).mean()

        latent_r = self.latent_regressor(r)

        kld = self.hparams.kld_weight * (-0.5 * (1 + logvar - (mu - latent_r) ** 2 - logvar.exp()).sum(dim=1)).mean()
        label_loss = self.hparams.label_weight * (-0.5 * (1 + logvar_r - (mu_r - gaze) ** 2 - logvar_r.exp()).sum(dim=1).mean())
        loss = label_loss + kld - recons_loss

        result = {"kld_loss": kld,
                  "label_loss": label_loss,
                  "recon_loss": -recons_loss,
                  "loss": loss}
        return result, reconstruction, img

    def training_step(self, batch, batch_idx):
        result, _, _ = self.shared_step(batch)
        train_result = {"train_" + k: v for k, v in result.items()}
        self.log_dict(train_result)
        return result["loss"]

    def validation_step(self, batch, batch_idx):
        result, recon, aug_img = self.shared_step(batch)
        valid_result = {"valid_" + k: v for k, v in result.items()}
        self.log_dict(valid_result)

        return result["loss"]

    def test_step(self, batch, batch_idx):
        result, _, _ = self.shared_step(batch)
        test_result = {"test_" + k: v for k, v in result.items()}
        self.log_dict(test_result)
        return result["loss"]

    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.RandomResizedCrop(size=(32, 32), scale=(0.5, 1.3)),
                                        transforms.RandomGrayscale(p=0.1),
                                        transforms.ColorJitter(brightness=0.5, hue=0.2, contrast=0.5, saturation=0.5),
                                        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.1),
                                        self._normaliser])
        _data = RTGENEH5Dataset(h5_pth=self.hparams.hdf5_file, subject_list=self._train_subjects, transform=transform)
        return DataLoader(_data, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_io_workers, pin_memory=True)

    def val_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(size=(32, 32)),
                                        self._normaliser])
        _data = RTGENEH5Dataset(h5_pth=self.hparams.hdf5_file, subject_list=self._validate_subjects, transform=transform)
        return DataLoader(_data, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_io_workers, pin_memory=True)

    def test_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(size=(32, 32)),
                                        self._normaliser])
        _data = RTGENEH5Dataset(h5_pth=self.hparams.hdf5_file, subject_list=self._test_subjects, transform=transform)
        return DataLoader(_data, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_io_workers, pin_memory=True)

    def configure_optimizers(self):
        def linear_warmup_decay(warmup_steps):
            def optimiser_schedule_fn(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                return 1.0

            return optimiser_schedule_fn

        params_to_update = []
        for name, param in itertools.chain(self.encoder.named_parameters(), self.decoder.named_parameters()):
            if param.requires_grad:
                params_to_update.append(param)

        optimizer = partial(OPTIMISERS[self.hparams.optimiser], params=params_to_update, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)()

        if self.hparams.optimiser_schedule:
            warmup_steps = (81152 / self.hparams.batch_size) * self.hparams.warmup_epochs
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, linear_warmup_decay(warmup_steps)),
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
        parser.add_argument('--weight_decay', default=1e-2, type=float)
        parser.add_argument('--encoder', choices=ENCODERS.keys(), default=list(ENCODERS.keys())[0])
        parser.add_argument('--decoder', choices=DECODERS.keys(), default=list(DECODERS.keys())[0])
        parser.add_argument('--optimiser_schedule', action="store_true", default=False)
        parser.add_argument('--warmup_epochs', type=int, default=10)
        parser.add_argument('--decay_reconstruction_loss', action="store_true", default=False)
        parser.add_argument('--optimiser', choices=OPTIMISERS.keys(), default=list(OPTIMISERS.keys())[0])
        parser.add_argument('--kld_weight', type=float, default=4e-2)
        parser.add_argument('--label_weight', type=float, default=1.0)
        parser.add_argument('--recon_weight', type=float, default=1e-3)
        parser.add_argument('--latent_dim', type=int, default=128)
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
    root_parser.add_argument('--eye_laterality', type=str, choices=["left", "right"], default="right")
    root_parser.add_argument('--num_io_workers', default=psutil.cpu_count(logical=False), type=int)
    root_parser.add_argument('--k_fold_validation', action="store_true", dest="k_fold_validation")
    root_parser.add_argument('--all_dataset', action='store_false', dest="k_fold_validation")
    root_parser.add_argument('--seed', type=int, default=0)
    root_parser.add_argument('--max_epochs', type=int, default=300, help="Maximum number of epochs to perform; the trainer will Exit after.")
    root_parser.add_argument('--distributed_strategy', choices=["none", "ddp_find_unused_parameters_false"], default="ddp_find_unused_parameters_false")
    root_parser.add_argument('--precision', choices=["16", "32"], default="32")
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
        callbacks = [ModelCheckpoint(monitor='valid_loss', mode='min', verbose=False, save_top_k=10, save_last=True,
                                     filename=f"fold={fold}" + "{epoch}-{valid_loss:.2f}"),
                     LearningRateMonitor(),
                     OnlineFineTuner(encoder_output_dim=512)]

        # start training
        trainer = Trainer(gpus=hyperparams.gpu,
                          strategy=None if hyperparams.distributed_strategy == "none" else hyperparams.distributed_strategy,
                          precision=int(hyperparams.precision),
                          callbacks=callbacks,
                          log_every_n_steps=10,
                          max_epochs=hyperparams.max_epochs)
        trainer.fit(model)
        # trainer.test()
