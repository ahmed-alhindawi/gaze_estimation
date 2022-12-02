import math

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from tqdm import tqdm
import os


class Gaze360FileLoader(data.Dataset):

    def __init__(self, root_path, file_name, fraction_total=1.0, transform=None, data_type=None):
        assert transform is not None
        assert data_type in ["training", "validation", "testing"]
        assert 0. < fraction_total <= 1.

        images = []
        with open(os.path.join(root_path, file_name), 'r') as f:
            lines = f.readlines()  # this reads all the lines, train.txt is ~10Mb so its okay
            req_lines = int(fraction_total * len(lines))
            lines_to_read = lines[:req_lines]
            for line in tqdm(lines_to_read, total=req_lines, desc=f"Loading {data_type} dataset"):
                line = line[:-1]
                line = line.replace("\t", " ")
                line = line.replace("  ", " ")
                split_lines = line.split(" ")
                if len(split_lines) > 3:
                    frame_number = int(split_lines[0].split('/')[-1][:-4])
                    for j in range(-3, 4):  # treat each image in the sequence independently
                        name_frame = '/'.join(split_lines[0].split('/')[:-1] + ['%0.6d.jpg' % (frame_number + j)])
                        name = os.path.join(root_path, name_frame)

                        gaze = np.array([float(split_lines[1]), float(split_lines[2]), float(split_lines[3])]).reshape((1, 3))
                        normalized_gaze = gaze / np.linalg.norm(gaze)
                        normalized_gaze = normalized_gaze.reshape(3)

                        spherical_vector = np.array([math.atan2(normalized_gaze[0], -normalized_gaze[2]), math.asin(normalized_gaze[1])])

                        item = (name, spherical_vector)
                        images.append(item)

        self.source_path = root_path
        self.file_name = file_name

        self.imgs = images
        self.transform = transform
        self.target_transform = transform
        self.loader = lambda x: Image.open(x).convert('RGB')

    def __getitem__(self, index):
        path_source, gaze = self.imgs[index]
        gaze_float = torch.Tensor(gaze)
        image = self.transform(self.loader(path_source))

        return image, gaze_float

    def __len__(self):
        return len(self.imgs)
