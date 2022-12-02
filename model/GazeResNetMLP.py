#! /usr/bin/env python
import torch.nn as nn
from torchvision import models

from gaze_estimation.model.AbstractModel import GazeEstimationAbstractModel


class GazeEstimationModelResNet(GazeEstimationAbstractModel):

    def __init__(self, num_out=2):
        super(GazeEstimationModelResNet, self).__init__()
        left_model = models.resnet18(pretrained=True)
        right_model = models.resnet18(pretrained=True)

        # remove the last ConvBRelu layer
        self.left_features = nn.Sequential(
            left_model.conv1,
            left_model.bn1,
            left_model.relu,
            left_model.maxpool,
            left_model.layer1,
            left_model.layer2,
            left_model.layer3,
            left_model.layer4,
            nn.Conv2d(512, 32, 1, 1)
        )

        self.right_features = nn.Sequential(
            right_model.conv1,
            right_model.bn1,
            right_model.relu,
            right_model.maxpool,
            right_model.layer1,
            right_model.layer2,
            right_model.layer3,
            right_model.layer4,
            nn.Conv2d(512, 32, 1, 1)
        )

        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        self.xl, self.xr, self.concat, self.fc1 = GazeEstimationAbstractModel.create_fc_layers(in_features=128, out_features=num_out)


if __name__ == "__main__":
    from tqdm import tqdm
    import torch
    model = GazeEstimationModelResNet()
    model.eval()
    model.to("cuda:0")

    d1 = torch.rand(1, 3, 36, 60).to("cuda:0")
    d2 = torch.rand(1, 3, 36, 60).to("cuda:0")
    d3 = torch.rand(1, 2).to("cuda:0")

    with torch.inference_mode():
        for _ in tqdm(range(10000)):
            _ = model(d1, d2, d3)
