import os

import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
import h5py


class RTGENEH5Dataset(data.Dataset):

    def __init__(self, h5_pth, subject_list=None, transform=None):
        self._h5_file = h5_pth
        self._transform = transform
        self._subject_labels = []

        assert subject_list is not None, "Must pass a list of subjects to load the data for"

        assert self._transform is not None
        self._base_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize((32, 32), transforms.InterpolationMode.NEAREST),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        wanted_subjects = ["s{:03d}".format(_i) for _i in subject_list]

        with h5py.File(h5_pth, mode="r") as h5_file:
            for grp_s_n in tqdm(wanted_subjects, desc="Loading subject metadata..."):  # subjects
                for grp_i_n, grp_i in h5_file[grp_s_n].items():  # images
                    if "left" in grp_i.keys() and "right" in grp_i.keys() and "label" in grp_i.keys():
                        left_dataset = grp_i["left"]
                        right_datset = grp_i['right']

                        assert len(left_dataset) == len(right_datset), "Dataset left/right images aren't equal length"
                        for _i in range(len(left_dataset)):
                            self._subject_labels.append(["/" + grp_s_n + "/" + grp_i_n, _i])

    def __len__(self):
        return len(self._subject_labels)

    def __getitem__(self, index):
        sample = self._subject_labels[index]

        with h5py.File(self._h5_file, mode="r") as h5_file:
            left_img = h5_file[sample[0] + "/left"][sample[1]][()]
            right_img = h5_file[sample[0] + "/right"][sample[1]][()]
            label_data = h5_file[sample[0] + "/label"][()]
            ground_truth_headpose = torch.Tensor(label_data[0][()].astype(np.float32))
            ground_truth_gaze = torch.Tensor(label_data[1][()].astype(np.float32))

            # Load data and get label
            transformed_left = self._transform(left_img)
            transformed_right = self._transform(right_img)
            base_left = self._base_transform(left_img)
            base_right = self._base_transform(right_img)

            return base_left, base_right, transformed_left, transformed_right, ground_truth_headpose, ground_truth_gaze
