import torch


class GNLL(torch.nn.Module):
    def __init__(self):
        super(GNLL, self).__init__()

    def forward(self, angle_out, label, var):
        loss = 0.5 * ((torch.exp(-var) * (angle_out - label) ** 2) + var)
        return loss.mean()