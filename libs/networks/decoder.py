import torch
import torch.nn as nn
from torch.nn import functional as F


class FPN(nn.Module):
    def __init__(self, inner_planes=256, out_planes=512, comb='cat', 
                 in_planes_list=[256, 512, 1024, 256]):
        super(FPN, self).__init__()
        self.comb = comb
        self.inner_planes = inner_planes
        self.out_planes = out_planes
        self.in_planes_list = in_planes_list

        assert comb in ['cat', 'sum'], (
            'The comb must be either \'cat\' nor \'sum\'')
        
        self.lats = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_planes, inner_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(inner_planes),
                nn.ReLU(inplace=True))
            for in_planes in in_planes_list[:-1]])

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(inner_planes, inner_planes//2, kernel_size=3, 
                          stride=1, padding=1, bias=False),
                nn.BatchNorm2d(inner_planes//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(inner_planes//2, inner_planes//2, kernel_size=3, 
                          stride=1, padding=1, bias=False),
                nn.BatchNorm2d(inner_planes//2),
                nn.ReLU(inplace=True))
            for _ in range(len(in_planes_list))])

        if comb == 'cat':
            bott_planes_in = inner_planes // 2 * len(in_planes_list)
        elif comb == 'sum':
            bott_planes_in = inner_planes // 2

        self.bottleneck = nn.Sequential(
            nn.Conv2d(bott_planes_in, out_planes, kernel_size=3, stride=1, 
                      padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

    def forward(self, xs):
        # Remove x1
        xs = xs[1:]
        for i in range(len(self.lats)):
            assert xs[i].size(1) == self.in_planes_list[i], (
                'The channel of xs[%d] is not matched with lats[%d]' % (i, i ))

        for i in range(len(xs) - 2, 0, -1):
            xs[i] = self.lats[i](xs[i])
            xs[i] = xs[i] + F.interpolate(xs[i+1], size=xs[i].size()[-2:],
                                          mode='bilinear', align_corners=True)

        for i in range(len(xs)):
            xs[i] = self.convs[i](xs[i])
            xs[i] = F.interpolate(xs[i], size=xs[0].size()[-2:], 
                                  mode='bilinear', align_corners=True)

        if self.comb == 'cat':
            x = torch.cat(xs, dim=1)
        elif self.comb == 'sum':
            x = sum(xs)
    
        x = self.bottleneck(x)
        return x



def test_FPN():
    xs = [torch.Tensor(1, 128, 257, 257),
          torch.Tensor(1, 256, 129, 129), torch.Tensor(1, 512, 65, 65),
          torch.Tensor(1, 1024, 65, 65), torch.Tensor(1, 256, 65, 65)]
    model = FPN(256, 512, 'cat')
    model.eval()
    print(model)
    x = model(xs)
    print(x.size())


if __name__ == '__main__':
    test_FPN()
