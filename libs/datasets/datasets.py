import importlib
from os import path as osp
import random

from abc import ABCmeta, abstractmethod
import cv2
import numpy as np

from torch.utils import data

import transforms


class BaseDataset(data.Dataset, metaclass=ABCmeta):
    def __init__(self, list_path, data_root, trans_config):
        super(BaseDataset, self).__init__()
        self.list_path = list_path
        self.data_root = data_root

        for name in trans_config['names']:
            assert name in transforms
        self.trans_config = trans_config

        self.file_list = self.read_list()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path_pair = self.file_list[idx]
        image, label = self.fetch_pair(path_pair)

        names = self.trans_config['names']
        configs = self.trans_config['configs']

        for name in names:
            transform = getattr(transforms, name)
            image, label = transform(image, label, configs.get(name, dict()))

        image, label = transforms.to_tensor(image, label)
        return image, label

    @abstractmethod
    def read_list(self):
        return None

    @abstractmethod
    def fetch_pair(self, path_pair):
        return None, None
        

class CityscapesDataset(BaseDataset):
    def __init__(self, list_path, data_root, trans_config):
        super(CityscapesDataset, self).__init__()
        self.id_table = {
            -1: ignore_label, 0: ignore_label, 1: ignore_label,
            2: ignore_label, 3: ignore_label, 4: ignore_label, 5: ignore_label,
            6: ignore_label, 7: 0, 8: 1, 9: ignore_label, 10: ignore_label,
            11: 2, 12: 3, 13: 4, 14: ignore_label, 15: ignore_label,
            16: ignore_label, 17: 5, 18: ignore_label, 19: 6, 20: 7, 21: 8,
            22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
            29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18,
        }
 
    def read_list(self):
        with open(self.list_path) as fp:
            lines = fp.readlines()
        file_list = [line.strip().split() for line in lines]
        return file_list

    def fetch_pair(self, path_pair):
        image_path = osp.join(self.data_root, path_pair[0])
        label_path = osp.join(self.data_root, path_pair[1])
        _, image_name = image_path.rsplit('/', 1)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        label = self.id2trainId(label)
        return image, label, image_name

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_table.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_table.items():
                label_copy[label == k] = v
        return label_copy


test_city_dst():
    list_path = './../datasets
