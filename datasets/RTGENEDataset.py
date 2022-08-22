import os

import numpy as np
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
from PIL import Image


class RTGENEFileDataset(data.Dataset):

    def __init__(self, root_path, subject_list=None, eye_transform=None, face_transform=None):
        self._root_path = root_path
        self._eye_transform = eye_transform
        self._face_transform = face_transform
        self._subject_labels = []

        assert subject_list is not None, "Must pass a list of subjects to load the data for"

        if self._eye_transform is None:
            self._eye_transform = transforms.Compose([transforms.Resize((36, 60), transforms.InterpolationMode.NEAREST),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        if self._face_transform is None:
            self._face_transform = transforms.Compose([transforms.Resize((224, 224), transforms.InterpolationMode.NEAREST),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        subject_path = [os.path.join(root_path, "s{:03d}_glasses/".format(_i)) for _i in subject_list]

        for subject_data in tqdm(subject_path, "Loading Subject metadata"):
            with open(os.path.join(subject_data, "label_combined.txt"), "r") as f:
                _lines = f.readlines()
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
        ground_truth_headpose = [sample[3], sample[4]]
        ground_truth_gaze = [sample[5], sample[6]]

        left_img = Image.open(os.path.join(self._root_path, sample[0])).convert('RGB')
        right_img = Image.open(os.path.join(self._root_path, sample[1])).convert('RGB')
        face_img = Image.open(os.path.join(self._root_path, sample[2])).convert('RGB')

        transformed_left = self._eye_transform(left_img)
        transformed_right = self._eye_transform(right_img)
        transformed_face = self._face_transform(face_img)
        #
        return np.asarray(left_img), np.asarray(right_img), np.asarray(face_img), \
               transformed_left, transformed_right, transformed_face, \
               np.array(ground_truth_headpose, dtype=np.float32), np.array(ground_truth_gaze, dtype=np.float32)
