import itertools
import os
from argparse import ArgumentParser

import h5py
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from gaze_estimation.datasets.RTGENEDataset import RTGENEH5Dataset, RTGENEFileDataset
from gaze_estimation.model import resnet18, decoder18, ProjectionHeadVAE
from gaze_estimation.training.LAMB import LAMB
from gaze_estimation.training.utils.OnlineLinearFinetuner import OnlineFineTuner


class TrainRTGENEAAVAE(pl.LightningModule):

    def __init__(self, hparams, train_subjects, validate_subjects, test_subjects, dataset):
        super(TrainRTGENEAAVAE, self).__init__()

        extract_img_fn = {
            "left": lambda x: (x[0], x[2]),
            "right": lambda x: (x[1], x[3])
        }

        self.encoder = resnet18()  # consider adding more backends
        self.projection = ProjectionHeadVAE(output_dim=hparams.latent_dim)
        self.decoder = decoder18(latent_dim=hparams.latent_dim)
        self._train_subjects = train_subjects
        self._validate_subjects = validate_subjects
        self._test_subjects = test_subjects
        self._extract_fn = extract_img_fn[hparams.eye_laterality]
        self._dataset_fn = dataset
        self.save_hyperparameters(hparams)

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
    def kl_divergence_mc(p, q, z):
        log_pz = p.log_prob(z)
        log_qz = q.log_prob(z)

        kl = (log_qz - log_pz).sum(dim=-1)

        return kl

    @staticmethod
    def kl_divergence_analytic(p, q, z):
        kl = torch.distributions.kl.kl_divergence(q, p).sum(dim=-1)

        return kl

    @staticmethod
    def sample(z_mu, z_var, eps=1e-6):
        std = torch.exp(z_var / 2.) + eps

        p = torch.distributions.Normal(torch.zeros_like(z_mu), torch.ones_like(std))
        q = torch.distributions.Normal(z_mu, std)
        z = q.rsample()

        return p, q, z

    def shared_step(self, batch):
        orig_img, augm_img = self._extract_fn(batch)
        encoding = self.forward(augm_img)
        mu, logvar = self.projection(encoding)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        reconstruction = self.decoder(z)

        recons_loss = self.gaussian_likelihood(reconstruction, sample=orig_img).mean()

        # kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        p, q, z = self.sample(mu, logvar)
        kld = self.kl_divergence_mc(p, q, z).mean()

        loss = self.hparams.kld_weight * kld - recons_loss
        result = {"kld_loss": kld,
                  "mse_loss": -recons_loss,
                  "loss": loss}
        return result, reconstruction, augm_img

    def training_step(self, batch, batch_idx):
        result, _, _ = self.shared_step(batch)
        train_result = {"train_" + k: v for k, v in result.items()}
        self.log_dict(train_result)
        return result["loss"]

    def validation_step(self, batch, batch_idx):
        result, recons, aug_imgs = self.shared_step(batch)
        valid_result = {"valid_" + k: v for k, v in result.items()}
        self.log_dict(valid_result)

        aug_grid = torchvision.utils.make_grid(aug_imgs[:8], normalize=True, scale_each=True)
        self.logger.experiment.add_image('aug_imgs', aug_grid, self.current_epoch)

        recon_grid = torchvision.utils.make_grid(recons[:8], normalize=True, scale_each=True)
        self.logger.experiment.add_image('reconstruction', recon_grid, self.current_epoch)

        return result["loss"]

    def test_step(self, batch, batch_idx):
        result, _, _ = self.shared_step(batch)
        test_result = {"test_" + k: v for k, v in result.items()}
        self.log_dict(test_result)
        return result["loss"]

    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.RandomResizedCrop(size=(32, 32), scale=(0.5, 1.2), interpolation=transforms.InterpolationMode.BILINEAR),
                                        transforms.RandomGrayscale(p=0.1),
                                        transforms.ColorJitter(brightness=0.5, hue=0.5, contrast=0.5, saturation=0.5),
                                        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.1),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
        _data = self._dataset_fn(root_path=self.hparams.dataset, subject_list=self._train_subjects, eye_transform=transform, data_fraction=0.95, data_type="training")
        return DataLoader(_data, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_io_workers, pin_memory=True)

    def val_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.RandomResizedCrop(size=(32, 32), scale=(0.5, 1.2), interpolation=transforms.InterpolationMode.BILINEAR),
                                        transforms.RandomGrayscale(p=0.1),
                                        transforms.ColorJitter(brightness=0.5, hue=0.5, contrast=0.5, saturation=0.5),
                                        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.1),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
        _data = self._dataset_fn(root_path=self.hparams.dataset, subject_list=self._validate_subjects, eye_transform=transform, data_fraction=0.05, data_type="validation")
        return DataLoader(_data, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_io_workers, pin_memory=True)

    def test_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(size=(32, 32)),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
        _data = self._dataset_fn(root_path=self.hparams.dataset, subject_list=self._test_subjects, eye_transform=transform, data_fraction=1.0, data_type="testing")
        return DataLoader(_data, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_io_workers, pin_memory=True)

    def configure_optimizers(self):
        def linear_warmup_decay(warmup_steps):
            """
            Linear warmup for warmup_steps, optionally with cosine annealing or
            linear decay to 0 at total_steps

            Adapted from grid_ai/vae
            """

            # this sucks
            def optimiser_schedule_fn(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))

                return 1.0  # do not decay after reaching the top

            return optimiser_schedule_fn

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
        parser.add_argument('--learning_rate', type=float, default=2.5e-4)
        parser.add_argument('--weight_decay', default=0, type=float)
        parser.add_argument('--optimiser_schedule', action="store_true", default=False)
        parser.add_argument('--warmup_epochs', type=int, default=10)
        parser.add_argument('--optimiser', choices=["adam_w", "lamb"], default="adam_w")
        parser.add_argument('--kld_weight', type=float, default=0)
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
    root_parser.add_argument('--dataset', type=str, default="rtgene_dataset.hdf5")
    root_parser.add_argument('--dataset_type', type=str, choices=["rt_gene", "other"], default="rt_gene")
    root_parser.add_argument('--eye_laterality', type=str, choices=["left", "right"], default="right")
    root_parser.add_argument('--num_io_workers', default=psutil.cpu_count(logical=False), type=int)
    root_parser.add_argument('--k_fold_validation', action="store_true", dest="k_fold_validation")
    root_parser.add_argument('--all_dataset', action='store_false', dest="k_fold_validation")
    root_parser.add_argument('--seed', type=int, default=0)
    root_parser.add_argument('--max_epochs', type=int, default=300, help="Maximum number of epochs to perform; the trainer will Exit after.")
    root_parser.add_argument('--distributed_strategy', choices=["none", "ddp_find_unused_parameters_false"], default="ddp_find_unused_parameters_false")
    root_parser.add_argument('--precision', choices=[16, 32], default=32, type=int)
    root_parser.set_defaults(k_fold_validation=True)

    model_parser = TrainRTGENEAAVAE.add_model_specific_args(root_parser)
    hyperparams = model_parser.parse_args()
    hyperparams.dataset = os.path.abspath(os.path.expanduser(hyperparams.dataset))

    if os.path.isfile(hyperparams.dataset):
        dataset_loader = RTGENEH5Dataset
    elif os.path.isdir(hyperparams.dataset):
        dataset_loader = RTGENEFileDataset
    else :
        raise ValueError("Unknown dataset as the dataset is neither a file (and therefore an HDF5) or a folder")

    print_args(hyperparams)

    pl.seed_everything(hyperparams.seed)

    train_subs = []
    valid_subs = []
    test_subs = []
    if hyperparams.dataset_type == "rt_gene":
        if hyperparams.k_fold_validation:
            train_subs.append([3, 4, 7, 9, 5, 6, 11, 12, 13, 14, 15, 16])
            train_subs.append([1, 2, 8, 10, 5, 6, 11, 12, 13, 14, 15, 16])
            train_subs.append([1, 2, 8, 10, 3, 4, 7, 9, 14, 15, 16])

            valid_subs.append([3, 4, 7, 9, 5, 6, 11, 12, 13, 14, 15, 16])
            valid_subs.append([1, 2, 8, 10, 5, 6, 11, 12, 13, 14, 15, 16])
            valid_subs.append([1, 2, 8, 10, 3, 4, 7, 9, 14, 15, 16])

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
        model = TrainRTGENEAAVAE(hparams=hyperparams, train_subjects=train_s, validate_subjects=valid_s, test_subjects=test_s, dataset=dataset_loader)

        # can't save valid_loss or valid_angle_loss as now that's a composite loss mediated by a training related parameter
        callbacks = [ModelCheckpoint(monitor='valid_loss', mode='min', verbose=False, save_top_k=10,
                                     filename="epoch={epoch}-valid_loss={valid_loss:.2f}", save_last=True),
                     LearningRateMonitor(),
                     OnlineFineTuner(encoder_output_dim=512, eye_laterality=hyperparams.eye_laterality)]

        # start training
        trainer = Trainer(accelerator="gpu",
                          devices=hyperparams.gpu,
                          strategy=None if hyperparams.distributed_strategy == "none" else hyperparams.distributed_strategy,
                          precision=int(hyperparams.precision),
                          callbacks=callbacks,
                          log_every_n_steps=10,
                          max_epochs=hyperparams.max_epochs)
        trainer.fit(model)
        # trainer.test()
