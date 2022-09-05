from torch import nn


class ProjectionHeadVAE(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super(ProjectionHeadVAE, self).__init__()

        self.first_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.mu = nn.Linear(hidden_dim, output_dim, bias=False)
        self.logvar = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        x = self.first_layer(x)
        return self.mu(x), self.logvar(x)


class ProjectionHeadVAERegression(nn.Module):

    def __init__(self, input_dim=512, hidden_dim=512, latent_dim=128, regressor_dim=2):
        super(ProjectionHeadVAERegression, self).__init__()

        self.first_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        self.mu = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.logvar = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.mu_r = nn.Linear(hidden_dim, regressor_dim, bias=False)
        self.logvar_r = nn.Linear(hidden_dim, regressor_dim, bias=False)

    def forward(self, x):
        x = self.first_layer(x)
        return self.mu(x), self.logvar(x), self.mu_r(x), self.logvar_r(x)
