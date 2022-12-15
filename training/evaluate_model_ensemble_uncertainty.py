import glob
import os
from argparse import ArgumentParser
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import reduce

from gaze_estimation.datasets.RTGENEDataset import RTGENEFileDataset as RTGENEDataset
from gaze_estimation.model import GazeEstimationModelVGG, GazeEstimationModelResNet
from utils.GazeAngleAccuracyMetric import GazeAngleAccuracyMetric

MODELS = {
    "vgg16": partial(GazeEstimationModelVGG, num_out=3),
    "resnet18": partial(GazeEstimationModelResNet, num_out=3)
}


def test_fold(d_loader, model_list, criterion, reduction_mode, model_idx="Ensemble"):
    assert type(model_list) is list, "model_list should be a list of models"
    angle_criterion_acc = []
    p_bar = tqdm(d_loader)
    for _, _, left, right, head_pose, gaze_labels in p_bar:
        p_bar.set_description(f"Testing  Model \"{model_idx}\"...")
        left = left.to("cuda:0")
        right = right.to("cuda:0")
        head_pose = head_pose.to("cuda:0")

        y = ([_m(left, right, head_pose).detach().cpu() for _m in model_list])
        y = torch.cat(y).reshape(len(model_list), -1, 3)
        raw_angle_out = y[:, :, :2]
        if reduction_mode == "average":
            angle_out = reduce(raw_angle_out[:, :, :2], "m b a -> b a", "mean")
        elif reduction_mode == "inverse_variance_weighted":
            var_out = 1.0 / reduce(1.0 / y[:, :, 2:].exp(), "m b a -> b a", "sum")
            angle_out = reduce(raw_angle_out / y[:, :, 2:].exp(), "m b a -> b a", "sum") * var_out
        else:
            raise ValueError("Unknown reduction method")
        angle_acc = criterion(angle_out, gaze_labels)

        angle_criterion_acc.extend(angle_acc.tolist())
        criterion.reset()

    angle_criterion_acc_arr = np.array(angle_criterion_acc)
    tqdm.write(f"\r\n\tModel: {model_idx}, Mean: {np.mean(angle_criterion_acc_arr)}, STD: {np.std(angle_criterion_acc_arr)}")


def load_model(path, model_base):
    model = MODELS.get(model_base)()
    model_prefix = "model."
    state_dict = {k[len(model_prefix):]: v for k, v in torch.load(path)['state_dict'].items() if k.startswith(model_prefix)}
    model.load_state_dict(state_dict)
    model = model.to("cuda:0")
    model = model.eval()
    return model


def main():
    root_parser = ArgumentParser(add_help=False)
    root_parser.add_argument('--model_loc', type=str, required=False, help='path to the model files to evaluate')
    root_parser.add_argument('--dataset_path', type=str, required=True)
    root_parser.add_argument('--num_io_workers', default=16, type=int)
    root_parser.add_argument('--model_base', choices=MODELS.keys(), default=list(MODELS.keys())[0])
    root_parser.add_argument('--batch_size', default=2, type=int)
    root_parser.add_argument('--reduction_mode', choices=["average", "inverse_variance_weighted"], default="inverse_variance_weighted")

    hyperparams = root_parser.parse_args()
    hyperparams.dataset_path = os.path.abspath(os.path.expanduser(hyperparams.dataset_path))
    hyperparams.model_loc = os.path.abspath(os.path.expanduser(hyperparams.model_loc))

    criterion = GazeAngleAccuracyMetric(reduction="none")

    # get each checkpoint and see which one i 1234
    # s best
    epoch_ckpt = glob.glob(os.path.abspath(os.path.join(hyperparams.model_loc, "*.ckpt")))
    if len(epoch_ckpt) <= 0:
        print("No checkpoints found")
        return

    models = [load_model(ckpt, hyperparams.model_base) for ckpt in epoch_ckpt]
    data_test = RTGENEDataset(root_path=hyperparams.dataset_path, data_type="testing", subject_list=[0], eye_transform=None, data_fraction=1.0)
    data_loader = DataLoader(data_test, batch_size=hyperparams.batch_size, shuffle=False, num_workers=hyperparams.num_io_workers, pin_memory=True)

    test_fold(data_loader, model_list=models, criterion=criterion, model_idx="ensemble", reduction_mode=hyperparams.reduction_mode)


if __name__ == "__main__":
    main()
