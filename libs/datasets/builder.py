import importlib
from base import BaseDataset
from cityscapes import CityscapesDataset


__dict__ = {
    'cityscapes': CityscapesDataset,
}


def build_dataset(dst_config):
    dst_cls = __dict__[dst_config['type']]
    dst = dst_cls(**dst_config.get('args', dict()))
    return dst
    
