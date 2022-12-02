import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
from pathlib import Path
import json
import math


class UnitGazeFileDataset(data.Dataset):

    def __init__(self, root_path: str, data_fraction: float, data_type="training", eye_transform=None, **kwargs):
        self._root_path = root_path
        self._eye_transform = eye_transform
        self._subject_labels = []

        assert data_type in ["training", "validation", "testing"]
        assert 0 < data_fraction <= 1

        if self._eye_transform is None:
            self._eye_transform = transforms.Compose([transforms.Resize((36, 60), transforms.InterpolationMode.NEAREST),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self._subject_labels = list(Path(root_path).resolve().glob("*.json"))

        if data_type == "training" or data_type == "testing":
            self._subject_labels = self._subject_labels[:math.floor(len(self._subject_labels) * data_fraction)]
        elif data_type == "validation":
            self._subject_labels = self._subject_labels[-math.floor(len(self._subject_labels) * data_fraction):]

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

            left = left_border[0]
            top = img.size[1] - ldmks_interior_margin[np.argmax(ldmks_interior_margin[:, 1]), 1].astype(int)
            right = right_border[0]
            bottom = img.size[1] - ldmks_interior_margin[np.argmin(ldmks_interior_margin[:, 1]), 1].astype(int)

            # zoom out a little
            left = int(left - 0.2 * horizontal_length)
            right = int(right + 0.2 * horizontal_length)
            top = int(top - 0.3 * horizontal_length)
            bottom = int(bottom + 0.3 * horizontal_length)

            img = img.crop((left, top, right, bottom))

            # convert look vector to polar coordinates
            look_vec = list(eval(data['eye_details']['look_vec']))[:3]
            gaze_theta, gaze_phi = np.arcsin(look_vec[1]), np.arctan2(-1 * look_vec[0], -1 * look_vec[2])
            gaze_angles = np.array([gaze_theta, gaze_phi]).astype(np.float32)

            # head_angle here is euler angles
            head_angles = np.array(list(eval(data['head_pose']))[:2])
            head_angles = np.deg2rad(head_angles)
            head_angles[0] = -head_angles[0]
            head_angles = head_angles.astype(np.float32)

        # let use a copy of the left and right images for now
        return torch.Tensor([0.0]), torch.Tensor([0.0]), self._eye_transform(img), self._eye_transform(img), gaze_angles, head_angles

    @staticmethod
    def process_json_list(json_list):
        return np.array([eval(s) for s in json_list])


if __name__ == "__main__":
    import cv2
    from tqdm import tqdm

    _transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize((36, 60), interpolation=transforms.InterpolationMode.NEAREST)])

    dataset = UnitGazeFileDataset(root_path="/home/ahmed/datasets/unity_eyes/", data_fraction=1.0, data_type="training", eye_transform=_transform)

    torch_to_pil = transforms.ToPILImage()

    for img, gaze, head in tqdm(dataset):
        img = torch_to_pil(img)
        img = np.asarray(img)[:, :, ::-1]
        img = np.ascontiguousarray(img)
        cv2.imshow("eye", img)
        cv2.waitKey(2000)

    cv2.destroyAllWindows()
