#! /usr/bin/env python

import torch.nn as nn
from torchvision import models
import torch


class GazeWithContextEstimationModelVGG(nn.Module):

    def __init__(self, num_out=2):
        super(GazeWithContextEstimationModelVGG, self).__init__()
        _left_model = models.vgg16(pretrained=True)
        _right_model = models.vgg16(pretrained=True)
        _face_model = models.vgg16(pretrained=True)

        # remove the last ConvBRelu layer
        _left_modules = [module for module in _left_model.features]
        _left_modules.append(_left_model.avgpool)
        self.left_features = nn.Sequential(*_left_modules)

        _right_modules = [module for module in _right_model.features]
        _right_modules.append(_right_model.avgpool)
        self.right_features = nn.Sequential(*_right_modules)

        _face_modules = [module for module in _face_model.features]
        _face_modules.append(_face_model.avgpool)
        self.face_features = nn.Sequential(*_face_modules)

        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True
        for param in self.face_features.parameters():
            param.requires_grad = True

        self.xl = nn.Sequential(
            nn.Linear(_left_model.classifier[0].in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU()
        )
        self.xr = nn.Sequential(
            nn.Linear(_right_model.classifier[0].in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU()
        )
        self.xf = nn.Sequential(
            nn.Linear(_face_model.classifier[0].in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU()
        )

        self.concat = nn.Sequential(
            nn.Linear(3072, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, num_out)
        )

    def forward(self, left_eye, right_eye, face):
        left_x = self.left_features(left_eye)
        left_x = torch.flatten(left_x, 1)
        left_x = self.xl(left_x)

        right_x = self.right_features(right_eye)
        right_x = torch.flatten(right_x, 1)
        right_x = self.xr(right_x)

        face_x = self.face_features(face)
        face_x = torch.flatten(face_x, 1)
        face_x = self.xf(face_x)

        context_x = torch.cat((left_x, right_x, face_x), dim=1)
        context_x = self.concat(context_x)

        fc1_output = self.fc1(context_x)
        fc2_output = self.fc2(fc1_output)
        return fc2_output


if __name__ == "__main__":
    left_eye = torch.rand(16, 3, 36, 60).float().to("cuda:0")
    right_eye = torch.rand(16, 3, 36, 60).float().to("cuda:0")
    face_eye = torch.rand(16, 3, 224, 224).float().to("cuda:0")

    model = GazeWithContextEstimationModelVGG()
    model.eval()
    model.to("cuda:0")

    out = model(left_eye, right_eye, face_eye)
    print(out)