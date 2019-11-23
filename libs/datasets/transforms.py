import os

import cv2
import numpy as np

__all__ = ['normalize', 'rand_resize', 'pad_border', 'rand_crop', 'rand_flip',
           'to_tensor']


def normalize(image, label, norm_config):
    if norm_config.get('use_rgb', False)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    scale = norm_config.get('scale', 1)
    mean = norm_config.get('mean', [0, 0, 0])
    std = norm_config.get('std', [1, 1, 1])

    image = image / scale
    image = (image - mean) / std
    return image, label


def rand_resize(image, label, resize_config):
    assert (image.shape[:2] == label.shape).all()

    min_scale = resize_config.get('min_scale', 0.5)
    max_scale = resize_config.get('max_scale', 2.0)
    scale = min_scale + random.rand() * (max_scale - min_scale)

    image = cv2.resize(image, None, fx=scale, fy=scale, 
                       interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, None, fx=scale, fy=scale, 
                       interpolation=cv2.INTER_NEAREST)
    return image, label


def pad_border(image, label, pad_config):
    assert (image.shape[:2] == label.shape).all()

    crop_h = pad_config.get('crop_h', 513)
    crop_w = pad_config.get('crop_w', 513)
    ignore_label = pad_config.get('ignore_label', -1)

    h, w = label.shape
    pad_h = max(crop_h - h, 0)
    pad_w = max(crop_w - w, 0)

    if pad_h > 0 or pad_w > 0:
        img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, 
            cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0))
        label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, 
            cv2.BORDER_CONSTANT, value=(ignore_label,))
    return image, label


def rand_crop(image, label, crop_config):
    assert (image.shape[:2] == label.shape).all()

    crop_h = pad_config.get('crop_h', 513)
    crop_w = pad_config.get('crop_w', 513)
    ignore_label = pad_config.get('ignore_label', -1)

    h, w = label.shape
    off_h = random.randint(0, h - crop_h)
    off_w = random.randint(0, w - crop_w)

    image = image[h_off: h_off+crop_h, w_off: w_off+crop_w]
    label = label[h_off: h_off+crop_h, w_off: w_off+crop_w]
    return image, label


def rand_flip(image, label, flip_config):
    flip_prob = flip_config.get('flip_prob', 0.5)
    if random.rand() < flip_prob:
        image = image[:, ::-1]
        label = label[:, ::-1]
    return image, label


def to_tensor(image, label):
    image = image.transpose(2, 0, 1)
    image = image.astype(np.float32)
    label = label.astype(np.int32)
    return image, label

