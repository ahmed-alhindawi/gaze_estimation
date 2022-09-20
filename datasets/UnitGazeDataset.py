import os

import math
import numpy as np
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import json


class UnitGazeFileDataset(data.Dataset):

    def __init__(self, root_path: str, data_fraction: float, data_type="training", eye_transform=None):
        self._root_path = root_path
        self._eye_transform = eye_transform
        self._subject_labels = []

        assert data_type in ["training", "validation", "testing"]
        assert 0 < data_fraction <= 1

        if self._eye_transform is None:
            self._eye_transform = transforms.Compose([transforms.Resize((36, 60), transforms.InterpolationMode.NEAREST),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self._orig_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize((32, 32)),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self._subject_labels = list(Path(root_path).resolve().glob("*.json"))

    def __len__(self):
        return len(self._subject_labels)

    def __getitem__(self, index):
        path = self._subject_labels[index]
        img_path = (Path(f"{path.parent}") / (path.name[:-5] + ".jpg")).resolve()
        img = Image.open(img_path)
        with open(path) as file:
            data = json.load(file)

            # need to crop onto the image
            ldmks_interior_margin = self.process_json_list(data['interior_margin_2d'])
            ldmks_caruncle = self.process_json_list(data['caruncle_2d'])

            # just for visualisation
            ldmks_iris = self.process_json_list(data['iris_2d'])
            eye_c = np.mean(ldmks_iris[:, :2], axis=0).astype(int)
            eye_c[1] = img.size[1] - eye_c[1]

            left_border = np.mean(ldmks_caruncle[:, :2], axis=0).astype(int)
            left_border[1] = img.size[1] - left_border[1]

            right_border = ldmks_interior_margin[np.argmax(ldmks_interior_margin[:, :1]), :2].astype(int)
            right_border[1] = img.size[1] - right_border[1]

            horizontal_length = np.linalg.norm(left_border - right_border).astype(int)

            left = int(left_border[0] - 0.1 * horizontal_length)
            top = img.size[1] - ldmks_interior_margin[np.argmax(ldmks_interior_margin[:, 1]), 1].astype(int)
            right = int(right_border[0] + 0.1 * horizontal_length)
            bottom = img.size[1] - ldmks_interior_margin[np.argmin(ldmks_interior_margin[:, 1]), 1].astype(int)

            img = img.crop((left, top, right, bottom))

            look_vec = list(eval(data['eye_details']['look_vec']))[:3]
            gaze_theta, gaze_phi = np.arcsin(look_vec[1]), np.arctan2(-1 * look_vec[0], -1 * look_vec[2])
            gaze_angles = np.array([gaze_theta, gaze_phi])

            head_angles = np.array(list(eval(data['head_pose']))[:2])
            head_angles = np.deg2rad(head_angles)
            head_angles[0] = -head_angles[0]

        return img, gaze_angles, head_angles, eye_c

    @staticmethod
    def process_json_list(json_list):
        return np.array([eval(s) for s in json_list])


if __name__ == "__main__":
    import cv2

    dataset = UnitGazeFileDataset(root_path="/home/ahmed/datasets/unity_eyes/", data_fraction=1.0, data_type="training")

    for img, gaze, head, eye_c in dataset:
        img = np.asarray(img)[:, :, ::-1]
        img = np.ascontiguousarray(img)

        cv2.imshow("eye", img)
        cv2.waitKey(2000)
