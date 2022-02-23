import torch
from torch import nn
import torch.nn.functional as F


class LatentRegressor(nn.Module):

    def __init__(self, input_dim=2, output_dim=128):
        super(LatentRegressor, self).__init__()

        self.first_layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        self.latent_regressor = nn.Linear(output_dim, output_dim, bias=False)

    def forward(self, x):
        x = self.first_layer(x)
        return self.latent_regressor(x)
