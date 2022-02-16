from functools import partial

import torch
from torch import nn
from torchvision import models
from enum import Enum


class GazeEncoder(nn.Module):

    class GazeEncoderBackend(Enum):
        Resnet18 = partial(models.resnet18, pretrained=True)
        Resnet34 = partial(models.resnet34, pretrained=True)
        Resnet50 = partial(models.resnet50, pretrained=True)

    def __init__(self, latent_dim: int = 128, backend: GazeEncoderBackend = GazeEncoderBackend.Resnet18) -> None:
        super(GazeEncoder, self).__init__()
        model = backend.value()
        self.encoder = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool
        )
        self.fc_mu = nn.Linear(model.fc.in_features, latent_dim)
        self.fc_var = nn.Linear(model.fc.in_features, latent_dim)

    def forward(self, img: torch.Tensor):
        result = self.encoder(img)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

