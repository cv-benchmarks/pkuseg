import torch
import torch.nn as nn

from utils import load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet50', 'renset101', 'resnet50_os16', 
           'resnet101_os16', 'resnet50_os8', 'resnet101_os8']


models_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet50-imagenet.pth',
    'resnet101': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth',
}


def conv1x1(in_planes, out_planes, stride=1):
    '''A 3x3 convolution with padding.'''
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    '''A 3x3 convolution with padding.'''
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     dilation=dilation, padding=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dilation=1, 
                 downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity 
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dilation=1, 
                 downsample=None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample

        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity 
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, deep_stem=True, dilation=8, use_mg=True):
        super(ResNet, self).__init__()
        self.deep_stem = deep_stem
        self.in_planes = 128 if deep_stem else 64
        self.out_planes = 2056
        multi_grid = [1, 2, 4] if use_mg else [1]

        if deep_stem:
            self.conv1 = conv3x3(3, 64, stride=2)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = conv3x3(64, 64)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = conv3x3(64, 128)
            self.bn3 = nn.BatchNorm2d(128)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        if dilation == 8:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, 
                                           dilation=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, 
                                           dilation=4, multi_grid=multi_grid)
        elif dilation == 16:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, 
                                           dilation=1)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, 
                                           dilation=2, multi_grid=multi_grid)
        elif dilation == 32:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, 
                                           dilation=1)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, 
                                           dilation=1, multi_grid=multi_grid)
        else:
            raise ValueError(
                'The dilation of ResNet must be one of 8, 16 and 32.')

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, 
                    multi_grid=[1]):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        grids = [multi_grid[i % len(multi_grid)] for i in range(blocks)]
        layers.append(
            block(self.in_planes, planes, stride, dilation=dilation*grids[0],
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.in_planes, planes, dilation=dilation*grids[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.deep_stem:
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x1 = self.relu(self.bn3(self.conv3(x)))
        else:
            x1 = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x1)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return [x1, x2, x3, x4, x5]


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(models_urls[arch],
                                              progress=progress)
        state_dict_for_load = dict()
        for name, param in model.named_parameters():
            if name in state_dict:
                state_dict_for_load[name] = state_dict[name]
        model.load_state_dict(state_dict_for_load)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    kwargs['deep_stem'] = False
    kwargs['dilation'] = 32
    kwargs['use_mg'] = False
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    kwargs['deep_stem'] = False
    kwargs['dilation'] = 32
    kwargs['use_mg'] = False
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    kwargs['dilation'] = 32
    kwargs['use_mg'] = False
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    kwargs['dilation'] = 32
    kwargs['use_mg'] = False
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet50_os16(pretrained=False, progress=True, **kwargs):
    kwargs['dilation'] = 16
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101_os16(pretrained=False, progress=True, **kwargs):
    kwargs['dilation'] = 16
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet50_os8(pretrained=False, progress=True, **kwargs):
    kwargs['dilation'] = 8
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101_os8(pretrained=False, progress=True, **kwargs):
    kwargs['dilation'] = 8
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


if __name__ == '__main__':
    img = torch.Tensor(1, 3, 224, 224)
    model = resnet50_os8(pretrained=False)
    model.eval()
    print(model)
    xs = model(img)
    for x in xs:
        print(x.size())
