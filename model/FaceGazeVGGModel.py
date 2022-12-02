#! /usr/bin/env python

import torch
import torch.nn as nn
import torchvision.models


class FaceGazeVGGModel(nn.Module):

    def __init__(self, num_out=2):
        super().__init__()
        _model = torchvision.models.vgg16(pretrained=True)

        # remove the last ConvBRelu layer
        model = [module for module in _model.features]
        model.append(nn.Conv2d(512, 1, 1, 1))
        model.append(nn.ReLU())
        model.append(nn.AvgPool2d(2, 2))
        self.model = nn.Sequential(*model)
        self.fc = nn.Linear(9, num_out, bias=True)

    def forward(self, face):
        x = self.model(face)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    from tqdm import tqdm

    model = FaceGazeVGGModel(nn_out=2)
    model.eval()
    model.to("cuda:0")

    face = torch.randn((1, 3, 244, 244)).float().cuda()

    with torch.inference_mode():
        for _ in tqdm(range(10000)):
            _ = model(face)
