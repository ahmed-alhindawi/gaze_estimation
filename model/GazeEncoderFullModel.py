import torch
from torch import nn


class GazeEncoderFullModel(nn.Module):
    @staticmethod
    def load_ckpt(model, ckpt):
        # the ckpt file saves the pytorch_lightning module which includes its child members. The only child member we're interested in is the "encoder".
        # Loading the state_dict with encoder creates an error as the model tries to find a child called encoder within it that doesn't
        # exist. Thus remove encoder from the dictionary and all is well.
        _model_prefix = "encoder."
        _state_dict = {k[len(_model_prefix):]: v for k, v in torch.load(ckpt)['state_dict'].items() if k.startswith(_model_prefix)}
        model.load_state_dict(_state_dict)
        return model

    def __init__(self, model_base, left_ckpt, right_ckpt, in_features=512, out_features=2):
        super(GazeEncoderFullModel, self).__init__()
        self.left_features = model_base()
        self.right_features = model_base()

        self.left_features = self.load_ckpt(self.left_features, left_ckpt)
        self.right_features = self.load_ckpt(self.right_features, right_ckpt)

        self.xl = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU()
        )
        self.xr = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU()
        )
        self.concat = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.GELU()

        )
        self.fc1 = nn.Sequential(
            nn.Linear(514, 256),
            nn.Tanhshrink()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, out_features)
        )

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
        fc2_output = self.fc2(fc1_output)

        return fc2_output