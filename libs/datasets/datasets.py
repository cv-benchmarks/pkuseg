import importlib
from os import path as osp
import random

from abc import ABCMeta, abstractmethod
import cv2
import numpy as np

from torch.utils import data

import transforms


class BaseDataset(data.Dataset, metaclass=ABCMeta):
    def __init__(self, list_path, data_root, trans_config):
        super(BaseDataset, self).__init__()
        self.list_path = list_path
        self.data_root = data_root

        for name in trans_config.get('names', []):
            assert name in transforms.__all__
        self.trans_config = trans_config

        self.file_list = self.read_list()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path_pair = self.file_list[idx]
        image, label = self.fetch_pair(path_pair)
        _, image_name = path_pair[0].rsplit('/', 1)

        names = self.trans_config.get('names', [])
        configs = self.trans_config.get('configs', dict())

        for name in names:
            transform = getattr(transforms, name)
            image, label = transform(image, label, configs.get(name, dict()))

        image, label = transforms.to_tensor(image, label)
        return image, label, image_name

    @abstractmethod
    def read_list(self):
        return None

    @abstractmethod
    def fetch_pair(self, path_pair):
        return None, None
        

class CityscapesDataset(BaseDataset):
    def __init__(self, list_path, data_root, trans_config):
        super(CityscapesDataset, self).__init__(list_path, data_root, 
                                                trans_config)
        ignore_label = trans_config.get('ignore_label', -1)
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

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        label = self.id2trainId(label)
        return image, label

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_table.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_table.items():
                label_copy[label == k] = v
        return label_copy


def test_city_dst():
    list_path = '../../data/cityscapes/train.lst'
    data_root = '/home1/lixia/Cityscapes'
    trans_config = {
        'names': ['normalize', 'rand_resize', 'pad_border', 'rand_crop', \
                  'rand_flip'],
        'configs': {
            'normalize': {
                'scale': 255,
                'mean': [0.48, 0.45, 0.42],
                'std': [0.27, 0.25, 0.22],
            },
            'rand_resize': {
                'min_scale': 0.75,
                'min_scale': 2.0,
            },
            'rand_crop': {
                'crop_h': 769,
                'crop_w': 769,
                'ignore_label': -1,
            },
            'pad_border': {
                'crop_h': 769,
                'crop_w': 769,
                'ignore_label': -1,
            },
        },
    }
    dataset = CityscapesDataset(list_path, data_root, trans_config)
    for i in range(10):
        image, label, name = dataset[i]
        print(image.shape, label.shape, name)
        print(image.mean(), image.std(), np.unique(label))


if __name__ == '__main__':
    test_city_dst()
