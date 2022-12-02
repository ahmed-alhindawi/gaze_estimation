#! /usr/bin/env python

import torch
import torch.nn as nn


class GazeEstimationAbstractModel(nn.Module):

    def __init__(self):
        super(GazeEstimationAbstractModel, self).__init__()

    @staticmethod
    def create_fc_layers(in_features, out_features):
        x_l = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.BatchNorm1d(16),
            nn.GELU()
        )
        x_r = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.BatchNorm1d(16),
            nn.GELU()
        )

        concat = nn.Sequential(
            nn.Linear(32, 18),
            nn.BatchNorm1d(18),
            nn.GELU()

        )

        fc1 = nn.Sequential(
            nn.Linear(20, out_features),
            nn.GELU()
        )

        return x_l, x_r, concat, fc1

    def forward(self, left_eye, right_eye, headpose):
        left_x = self.left_features(left_eye)
        left_x = torch.flatten(left_x, 1)
        left_x = self.xl(left_x)

        right_x = self.right_features(right_eye)
        right_x = torch.flatten(right_x, 1)
        right_x = self.xr(right_x)

        eyes_x = torch.cat((left_x, right_x), dim=1)
        eyes_x = self.concat(eyes_x)

        eyes_headpose = torch.cat((eyes_x, headpose), dim=1)

        fc1_output = self.fc1(eyes_headpose)

        return fc1_output
