import glob
import os
from argparse import ArgumentParser
from functools import partial

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from gaze_estimation.datasets.RTGENEDataset import RTGENEFileDataset as RTGENEDataset
from gaze_estimation.model import GazeEstimationModelVGG, GazeEstimationModelResNet
from utils.GazeAngleAccuracyMetric import GazeAngleAccuracyMetric

MODELS = {
    "vgg16": partial(GazeEstimationModelVGG, num_out=3),
    "resnet18": partial(GazeEstimationModelResNet, num_out=3)
}


def test_fold(d_loader, model_list, fold_idx, criterion, mean_log_variance, reduction_mode, softmax_temperature, model_idx="Ensemble"):
    assert type(model_list) is list, "model_list should be a list of models"
    angle_criterion_acc = []
    p_bar = tqdm(d_loader)
    for _, _, left, right, head_pose, gaze_labels in p_bar:
        p_bar.set_description("Testing Fold {}, Model \"{}\"...".format(fold_idx, model_idx))
        left = left.to("cuda:0")
        right = right.to("cuda:0")
        head_pose = head_pose.to("cuda:0")

        y = ([_m(left, right, head_pose)[0].detach().cpu() for _m in model_list])
        y = torch.cat(y).reshape(len(model_list), -1, 3)
        raw_angle_out = y[:, :, :2]
        if reduction_mode == "average":
            angle_out = torch.mean(raw_angle_out[:, :, :2], dim=0)
        elif reduction_mode == "weighted_average":
            log_var_out = -y[:, :, 2] + mean_log_variance
            softmax_log_var_out = torch.softmax(log_var_out/softmax_temperature, dim=0).unsqueeze(-1)

            angle_out = torch.sum(raw_angle_out * softmax_log_var_out, dim=0)
        else:
            raise ValueError("Unknown reduction method")
        angle_acc = criterion(angle_out, gaze_labels)

        angle_criterion_acc.extend(angle_acc.tolist())
        criterion.reset()

    angle_criterion_acc_arr = np.array(angle_criterion_acc)
    tqdm.write(f"\r\n\tFold: {fold_idx}, Model: {model_idx}, Mean: {np.mean(angle_criterion_acc_arr)}, STD: {np.std(angle_criterion_acc_arr)}")


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
    root_parser.add_argument('--mean_log_variance', default=-11.0725, type=float)
    root_parser.add_argument('--softmax_temperature', default=1, type=float)
    root_parser.add_argument('--reduction_mode', choices=["average", "weighted_average"], default="weighted_average")

    hyperparams = root_parser.parse_args()
    hyperparams.dataset_path = os.path.abspath(os.path.expanduser(hyperparams.dataset_path))
    hyperparams.model_loc = os.path.abspath(os.path.expanduser(hyperparams.model_loc))

    test_subjects = [[1, 2, 8, 10], [3, 4, 7, 9], [5, 6, 11, 12, 13]]
    criterion = GazeAngleAccuracyMetric(reduction="none")

    eye_transform = transforms.Compose([transforms.Resize((36, 60), transforms.InterpolationMode.NEAREST),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    folds = [os.path.abspath(os.path.join(hyperparams.model_loc, f"fold_{i}/")) for i in range(3)]
    for fold_idx, (test_subject, fold) in enumerate(zip(test_subjects, folds)):
        # get each checkpoint and see which one is best
        epoch_ckpt = glob.glob(os.path.abspath(os.path.join(fold, "*.ckpt")))
        if len(epoch_ckpt) <= 0:
            continue

        models = [load_model(ckpt, hyperparams.model_base) for ckpt in epoch_ckpt]
        data_test = RTGENEDataset(root_path=hyperparams.dataset_path, subject_list=test_subject, eye_transform=eye_transform)
        data_loader = DataLoader(data_test, batch_size=hyperparams.batch_size, shuffle=False, num_workers=hyperparams.num_io_workers, pin_memory=True)

        test_fold(data_loader, model_list=models, fold_idx=fold_idx, criterion=criterion, model_idx="ensemble", mean_log_variance=hyperparams.mean_log_variance,
                  reduction_mode=hyperparams.reduction_mode, softmax_temperature=hyperparams.softmax_temperature)


if __name__ == "__main__":
    main()
