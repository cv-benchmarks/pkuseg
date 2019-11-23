import torch
import torch.nn as nn
from torch.nn import functional as F


class FCNHead(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(FCNHead, self).__init__()
        self.out_planes = out_planes
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, 
                      padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.bottleneck(x)


class ASPP(nn.Module):
    def __init__(self, in_planes, inner_planes=256, out_planes=512, 
                 dilations=(12, 24, 36)):
        super(ASPP, self).__init__()
        self.out_planes = out_planes

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, inner_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_planes),
            nn.ReLU())
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, inner_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_planes), nn.ReLU())
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_planes, inner_planes, kernel_size=3, 
                          padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(inner_planes), nn.ReLU())
            for dilation in dilations])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_planes*(len(dilations)+2), out_planes, 
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

    def forward(self, x):
        _, _, h, w = x.size()

        glb = self.gap(x)
        glb = F.interpolate(glb, size=(h, w), mode='bilinear', 
                            align_corners=True)
        outs = [self.conv(x), glb]
        
        for dilated_conv in self.dilated_convs:
            outs.append(dilated_conv(x))

        out = torch.cat(outs, 1)
        out = self.bottleneck(out)
        return out


if __name__ == '__main__':
    img = torch.Tensor(1, 2048, 65, 65)
    model = ASPP(2048, 256, 512)
    model.eval()
    print(model)
    x = model(img)
    print(x.size())
