#! /usr/bin/env python

import torch.nn as nn
from torchvision import models

from gaze_estimation.model.AbstractModel import GazeEstimationAbstractModel


class GazeEstimationModelVGG(GazeEstimationAbstractModel):

    def __init__(self, hidden_dim=32, num_out=2):
        super(GazeEstimationModelVGG, self).__init__()
        _left_model = models.vgg16(pretrained=True)
        _right_model = models.vgg16(pretrained=True)

        # remove the last ConvBRelu layer
        _left_modules = [module for module in _left_model.features]
        _left_modules.append(nn.Conv2d(512, hidden_dim, 1, 1))
        self.left_features = nn.Sequential(*_left_modules)

        _right_modules = [module for module in _right_model.features]
        _right_modules.append(nn.Conv2d(512, hidden_dim, 1, 1))
        self.right_features = nn.Sequential(*_right_modules)

        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        self.xl, self.xr, self.concat, self.fc1 = GazeEstimationAbstractModel.create_fc_layers(in_features=hidden_dim, out_features=num_out)


if __name__ == "__main__":
    from tqdm import tqdm
    import torch
    model = GazeEstimationModelVGG()
    model.eval()
    model.to("cuda:0")

    d1 = torch.rand(1, 3, 36, 60).to("cuda:0")
    d2 = torch.rand(1, 3, 36, 60).to("cuda:0")
    d3 = torch.rand(1, 2).to("cuda:0")

    with torch.inference_mode():
        for _ in tqdm(range(10000)):
            _ = model(d1, d2, d3)
