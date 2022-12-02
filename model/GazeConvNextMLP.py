import timm
import torch
from gaze_estimation.model.AbstractModel import GazeEstimationAbstractModel


class GazeEstimationModelConvNeXt(GazeEstimationAbstractModel):

    def __init__(self, num_out=2):
        super(GazeEstimationModelConvNeXt, self).__init__()
        left_model = timm.create_model("convnext_small", pretrained=True, features_only=True)
        right_model = timm.create_model("convnext_small", pretrained=True, features_only=True)

        self.left_features = left_model
        self.right_features = right_model

        self.xl, self.xr, self.concat, self.fc1 = GazeEstimationAbstractModel.create_fc_layers(in_features=768, out_features=num_out)

    def forward(self, left_eye, right_eye, headpose):
        left_x = self.left_features(left_eye)[3]
        left_x = torch.flatten(left_x, 1)
        left_x = self.xl(left_x)

        right_x = self.right_features(right_eye)[3]
        right_x = torch.flatten(right_x, 1)
        right_x = self.xr(right_x)

        eyes_x = torch.cat((left_x, right_x), dim=1)
        eyes_x = self.concat(eyes_x)

        eyes_headpose = torch.cat((eyes_x, headpose), dim=1)

        fc1_output = self.fc1(eyes_headpose)

        return fc1_output
