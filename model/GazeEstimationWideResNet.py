import timm
import torch
from .AbstractModel import GazeEstimationAbstractModel


class GazeEstimationWideResNet(GazeEstimationAbstractModel):

    def __init__(self, num_out=2):
        super(GazeEstimationWideResNet, self).__init__()
        left_model = timm.create_model("wide_resnet50_2", pretrained=True, features_only=True)
        right_model = timm.create_model("wide_resnet50_2", pretrained=True, features_only=True)

        self.left_features = left_model
        self.right_features = right_model

        self.xl, self.xr, self.concat, self.fc1 = GazeEstimationAbstractModel.create_fc_layers(in_features=12288, out_features=num_out)

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


if __name__ == "__main__":
    from tqdm import tqdm
    model = GazeEstimationWideResNet()
    model.eval()
    model.to("cuda:0")

    d1 = torch.rand(1, 3, 36, 60).to("cuda:0")
    d2 = torch.rand(1, 3, 36, 60).to("cuda:0")
    d3 = torch.rand(1, 2).to("cuda:0")

    for _ in tqdm(range(100)):
        _ = model(d1, d2, d3)
