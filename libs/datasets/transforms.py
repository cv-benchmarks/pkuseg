import os

import cv2
import numpy as np
from numpy import random


__all__ = ['normalize', 'rand_resize', 'pad_border', 'rand_crop', 'rand_flip',
           'to_tensor']


def normalize(image, label, mean, std, scale=1, use_rgb=False):
    if use_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / scale
    image = (image - mean) / std
    return image, label


def rand_resize(image, label, min_scale=0.5, max_scale=2.0):
    assert image.shape[:2] == label.shape
    scale = min_scale + random.rand() * (max_scale - min_scale)
    image = cv2.resize(image, None, fx=scale, fy=scale, 
                       interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, None, fx=scale, fy=scale, 
                       interpolation=cv2.INTER_NEAREST)
    return image, label


def pad_border(image, label, crop_h, crop_w, ignore_index=-1):
    assert image.shape[:2] == label.shape
    h, w = label.shape
    pad_h = max(crop_h - h, 0)
    pad_w = max(crop_w - w, 0)

    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, 
                                   cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0))
        label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, 
                                   cv2.BORDER_CONSTANT, value=(ignore_index,))
    return image, label


def rand_crop(image, label, crop_h, crop_w, ignore_index=-1):
    assert image.shape[:2] == label.shape
    h, w = label.shape
    off_h = random.randint(0, max(h - crop_h, 0) + 1)
    off_w = random.randint(0, max(w - crop_w, 0) + 1)

    image = image[off_h: off_h+crop_h, off_w: off_w+crop_w]
    label = label[off_h: off_h+crop_h, off_w: off_w+crop_w]
    return image, label


def rand_flip(image, label, flip_prob=0.5):
    if random.rand() < flip_prob:
        image = image[:, ::-1]
        label = label[:, ::-1]
    return image, label


def to_tensor(image, label):
    image = image.transpose(2, 0, 1)
    image = image.astype(np.float32)
    label = label.astype(np.int32)
    return image, label

