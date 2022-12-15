#! /usr/bin/env python

import torch.nn as nn
import timm

from gaze_estimation.model.AbstractModel import GazeEstimationAbstractModel


class GazeEstimationModelEfficientFormer(GazeEstimationAbstractModel):

    def __init__(self, num_out=2):
        super(GazeEstimationModelEfficientFormer, self).__init__()
        _left_model = timm.create_model("efficientformer_l1", pretrained=True, in_chans=3, num_classes=0)
        _right_model = timm.create_model("efficientformer_l1", pretrained=True, in_chans=3, num_classes=0)

        self.left_features = _left_model
        self.right_features = _right_model

        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        self.xl, self.xr, self.concat, self.fc1 = GazeEstimationAbstractModel.create_fc_layers(in_features=448, out_features=num_out)


if __name__ == "__main__":
    from tqdm import tqdm
    import torch
    model = GazeEstimationModelEfficientFormer()
    model.eval()
    model.to("cuda:0")

    d1 = torch.rand(1, 3, 224, 224).to("cuda:0")
    d2 = torch.rand(1, 3, 224, 224).to("cuda:0")
    d3 = torch.rand(1, 2).to("cuda:0")

    with torch.inference_mode():
        for _ in tqdm(range(10000)):
            _ = model(d1, d2, d3)
