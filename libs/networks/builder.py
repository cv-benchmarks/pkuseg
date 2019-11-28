import importlib

import torch
from torch import nn
from torch.nn import functional as F

from heads import FCNHead


class ModelBuilder(nn.Module):
    def __init__(self, net_config, aux_config=None):
        super(ModelBuilder, self).__init__()

        num_classes = net_config['num_classes']
        out_planes = int(net_config['out_planes'])
        self.use_aux = aux_config is not None

        if self.use_aux:
            in_planes = int(aux_config['in_planes'])
            out_planes = int(aux_config['out_planes'])
            self.aux = FCNHead(in_planes, out_planes)
            self.aux_clsf = nn.Conv2d(out_planes, num_classes)

        self.encoder = self._build_module(net_config, 'encoder')
        self.seg_head = self._build_module(net_config, 'seg_head')
        self.decoder = self._build_module(net_config, 'decoder')
        assert self.encoder is not None, 'There must be an encoder!'

        if out_planes >= 256:
            self.clsf = nn.Sequential(
                nn.Dropout2d(0.1),
                nn.Conv2d(out_planes, num_classes, 1))
        else:
            self.clsf = nn.Conv2d(out_planes, num_classes)


    def _build_module(self, net_config, key):
        cls_config = net_config.get(key, None)
        if cls_config is None:
            return None

        cls_type = cls_config['type']
        mod_name, cls_name = cls_type.rsplit('.', 1)
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        return cls(**cls_config.get('args', dict()))

    def forward(self, x):
        h, w = x.size()[-2:]
        xs = self.encoder(x)

        if self.seg_head is not None:
            xs[-1] = self.seg_head(xs[-1])

        if self.decoder is not None:
            x = self.decoder(xs)
        else:
            x = xs[-1]

        pred = self.clsf(x)
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', 
                             align_corners=True)

        if not self.use_aux:
            return [pred, None]

        aux = self.aux(xs[-2])
        aux = self.aux_clsf(aux)
        aux = F.interpolate(aux, size=(h, w), mode='bilinear', 
                            align_corners=True)

        return [pred, aux]


if __name__ == '__main__':
    net_config = {
        'num_classes': 19,
        'out_planes': 512,
        'encoder': {
            'type': 'resnet.resnet50',
        },
        'seg_head': {
            'type': 'heads.ASPP',
            'args': {
                'in_planes': 2048,
            }
        },
    }
    model = ModelBuilder(net_config)
    print(model)
    input = torch.Tensor(2, 3, 513, 513)
    outputs = model(input)
    for output in outputs:
        if output is not None:
            print(output.size())

