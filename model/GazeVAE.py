import torch
from torch import nn


class GazeEncoder(nn.Module):

    def __init__(self, in_channels: int = 3, latent_dim: int = 128) -> None:
        super(GazeEncoder, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        hidden_dims = [in_channels, 32, 64, 128, 256, 512]

        # Build Encoder
        for curr_hid, next_hid in zip(hidden_dims, hidden_dims[1:]):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(curr_hid, out_channels=next_hid, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(next_hid),
                    nn.GELU())
            )

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

    def forward(self, img: torch.Tensor):
        result = self.encoder(img)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var


class GazeDecoder(nn.Module):

    def __init__(self, latent_dim: int=128):
        super(GazeDecoder, self).__init__()
        # Build Decoder
        modules = []
        hidden_dims = [32, 64, 128, 256, 512]
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for curr_dim, next_dim in zip(hidden_dims, hidden_dims[1:]):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(curr_dim, next_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(next_dim),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_input(z)
        x = x.view(-1, 512, 2, 2)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x


if __name__ == "__main__":
    enco = GazeEncoder(in_channels=3, latent_dim=128)
    deco = GazeDecoder(latent_dim=128)
    x = torch.rand(1, 3, 64, 64)
    mu, logvar = enco(x)

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = eps * std + mu
    y = deco(z)
    print(y.shape)
