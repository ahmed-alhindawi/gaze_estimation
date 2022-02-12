#! /usr/bin/env python

import torch.nn as nn
from torchvision import models

from .AbstractModel import GazeEstimationAbstractModel


class GazeEstimationModelVGG(GazeEstimationAbstractModel):

    def __init__(self, num_out=2):
        super(GazeEstimationModelVGG, self).__init__()
        _left_model = models.vgg16(pretrained=True)
        _right_model = models.vgg16(pretrained=True)

        # remove the last ConvBRelu layer
        _left_modules = [module for module in _left_model.features]
        _left_modules.append(_left_model.avgpool)
        self.left_features = nn.Sequential(*_left_modules)

        _right_modules = [module for module in _right_model.features]
        _right_modules.append(_right_model.avgpool)
        self.right_features = nn.Sequential(*_right_modules)

        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        self.xl, self.xr, self.concat, self.fc1, self.fc2 = GazeEstimationAbstractModel.create_fc_layers(in_features=_left_model.classifier[0].in_features,
                                                                                                         out_features=num_out)
