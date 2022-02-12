#! /usr/bin/env python

import torch.nn as nn
from torchvision import models

from .AbstractModel import GazeEstimationAbstractModel


class GazeEstimationModelResNet(GazeEstimationAbstractModel):

    def __init__(self, num_out=2,):
        super(GazeEstimationModelResNet, self).__init__()
        _left_model = models.resnet18(pretrained=True)
        _right_model = models.resnet18(pretrained=True)

        # remove the last ConvBRelu layer
        self.left_features = nn.Sequential(
            _left_model.conv1,
            _left_model.bn1,
            _left_model.relu,
            _left_model.maxpool,
            _left_model.layer1,
            _left_model.layer2,
            _left_model.layer3,
            _left_model.layer4,
            _left_model.avgpool
        )

        self.right_features = nn.Sequential(
            _right_model.conv1,
            _right_model.bn1,
            _right_model.relu,
            _right_model.maxpool,
            _right_model.layer1,
            _right_model.layer2,
            _right_model.layer3,
            _right_model.layer4,
            _right_model.avgpool
        )

        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        self.xl, self.xr, self.concat, self.fc1, self.fc2 = GazeEstimationAbstractModel.create_fc_layers(in_features=_left_model.fc.in_features, out_features=num_out)

