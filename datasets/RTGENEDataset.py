import os

import math
import numpy as np
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import h5py


class RTGENEFileDataset(data.Dataset):

    def __init__(self, root_path, data_fraction: float, subject_list=None, data_type="training", eye_transform=None):
        self._root_path = root_path
        self._eye_transform = eye_transform
        self._subject_labels = []

        assert subject_list is not None, "Must pass a list of subjects to load the data for"
        assert data_type in ["training", "validation", "testing"]
        assert 0 < data_fraction <= 1

        if self._eye_transform is None:
            self._eye_transform = transforms.Compose([transforms.Resize((36, 60), transforms.InterpolationMode.NEAREST),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self._orig_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize((32, 32)),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        subject_path = [os.path.join(root_path, "s{:03d}_glasses/".format(_i)) for _i in subject_list]

        for subject_data in tqdm(subject_path, "Loading Subject metadata"):
            with open(os.path.join(subject_data, "label_combined.txt"), "r") as f:
                _lines = f.readlines()
                if data_type == "training" or data_type == "testing":
                    _lines = _lines[:math.floor(len(_lines) * data_fraction)]
                elif data_type == "validation":
                    _lines = _lines[-math.floor(len(_lines) * data_fraction):]

                for line in _lines:
                    split = line.split(",")
                    left_img_path = os.path.join(subject_data, "inpainted/left/", "left_{:0=6d}_rgb.png".format(int(split[0])))
                    right_img_path = os.path.join(subject_data, "inpainted/right/", "right_{:0=6d}_rgb.png".format(int(split[0])))
                    face_img_path = os.path.join(subject_data, "inpainted/face/", "face_{:0=6d}_rgb.png".format(int(split[0])))
                    if os.path.exists(left_img_path) and os.path.exists(right_img_path):
                        head_phi = float(split[1].strip()[1:])
                        head_theta = float(split[2].strip()[:-1])
                        gaze_phi = float(split[3].strip()[1:])
                        gaze_theta = float(split[4].strip()[:-1])
                        self._subject_labels.append([left_img_path, right_img_path, face_img_path, head_phi, head_theta, gaze_phi, gaze_theta])

    def __len__(self):
        return len(self._subject_labels)

    def __getitem__(self, index):
        sample = self._subject_labels[index]
        ground_truth_headpose = np.array([sample[3], sample[4]], dtype=np.float32)
        ground_truth_gaze = np.array([sample[5], sample[6]], dtype=np.float32)

        left_img = Image.open(os.path.join(self._root_path, sample[0])).convert('RGB')
        right_img = Image.open(os.path.join(self._root_path, sample[1])).convert('RGB')

        transformed_left = self._eye_transform(left_img)
        transformed_right = self._eye_transform(right_img)

        return self._orig_transform(left_img), self._orig_transform(right_img), transformed_left, transformed_right, ground_truth_headpose, ground_truth_gaze


class RTGENEH5Dataset(data.Dataset):

    def __init__(self, root_path, data_fraction: float, subject_list=None, data_type="training", eye_transform=None):
        self._h5_file = root_path
        self._transform = eye_transform
        self._subject_labels = []

        assert subject_list is not None, "Must pass a list of subjects to load the data for"
        assert data_type in ["training", "validation", "testing"]
        assert 0 < data_fraction <= 1

        if self._transform is None:
            self._transform = transforms.Compose([transforms.Resize((36, 60)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self._orig_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize((32, 32)),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        wanted_subjects = ["s{:03d}".format(_i) for _i in subject_list]

        with h5py.File(self._h5_file, mode="r") as dataset:
            for subject_ids in tqdm(wanted_subjects, desc="Loading subject metadata..."):  # subjects
                images = list(dataset[subject_ids].items())
                if data_type == "training" or data_type == "testing":
                    images = images[:math.floor(len(images) * data_fraction)]
                elif data_type == "validation":
                    images = images[-math.floor(len(images) * data_fraction):]

                for _, subject_img_grp in images:
                    if "left" in subject_img_grp.keys() and "right" in subject_img_grp.keys() and "label" in subject_img_grp.keys():
                        self._subject_labels.append(subject_img_grp.name)

        assert len(self._subject_labels) > 0

    def __len__(self):
        return len(self._subject_labels)

    def __getitem__(self, index):
        sample = self._subject_labels[index]

        with h5py.File(self._h5_file, mode="r") as dataset:
            left_img = dataset[sample + "/left"][0][()]
            right_img = dataset[sample + "/right"][0][()]
            label_data = dataset[sample + "/label"][()]

        ground_truth_headpose = label_data[0][()].astype(np.float32)
        ground_truth_gaze = label_data[1][()].astype(np.float32)

        # Load data and get label
        transformed_left = self._transform(Image.fromarray(left_img, 'RGB'))
        transformed_right = self._transform(Image.fromarray(right_img, 'RGB'))

        return self._orig_transform(left_img), self._orig_transform(right_img), transformed_left, transformed_right, ground_truth_headpose, ground_truth_gaze

