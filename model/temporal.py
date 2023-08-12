# Code for "An End-to-End Two-Branch Network Towards Robust Video Fingerprinting"
# Yingying Xu, Yuanding Zhou, Xinran Li, Gejian Zhao, and Chuan Qin

import torch
import torch.nn as nn
import torch.nn.functional as F

class TAB(nn.Module):
    def __init__(self,in_channels,kernel_size=3,stride=1,padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.L = nn.Sequential(
            nn.Conv1d(in_channels,
                      in_channels // 4,
                      kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      bias=False), nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // 4, in_channels, kernel_size, stride=1, padding=2, dilation=2, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        n, c, t, h, w = x.size()
        identity = x

        # out:(n*c,t,1,1)
        out = F.adaptive_avg_pool2d(x.view(n * c, t, h, w), (1, 1))

        # local_activation：(n,c,t,1,1)
        local_activation = self.L(out.view(n,c,t)).view(n, c, t, 1, 1)

        # new_x1:(n,c,t,h,w)
        new_x1 = x * local_activation
        final = identity + new_x1

        return final

class BottleNeckt(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, t, (3,1,1), stride=(1,1,1), padding=(1,0,0), bias=False),
            nn.BatchNorm3d(t),
            nn.ReLU(inplace=True),

            nn.Conv3d(t, t, (1, 3, 3), stride=stride, padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(t),
            nn.ReLU(inplace=True),

            nn.Conv3d(t, out_channels, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.shortcut = nn.Sequential()
        if stride == (1,1,1) and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, (1,1,1), stride=(1,1,1), padding=(0,0,0), bias=False),
                nn.BatchNorm3d(out_channels),
            )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x1 = self.conv(x)

        if self.stride == (1,1,1):
            return x1 + self.shortcut(x)
        else:
            return x1


class Temporalpath(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 8, (5,7,7), stride=(1,2,2), padding=(2,3,3), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True)
        )
        self.bottleneckt1 = self.make_layer(1, 8, 32, (1,1,1), 8)
        self.bottleneckt2 = self.make_layer(1, 32, 64, (1,2,2), 16)
        self.bottleneckt3 = self.make_layer(1, 64, 128, (1,2,2), 32)
        self.bottleneckt4 = self.make_layer(1, 128, 256, (1,2,2), 64)
        self.globalavgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.tab1 = TAB(32)
        self.tab2 = TAB(64)
        self.tab3 = TAB(128)
        self.tab4 = TAB(256)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    def make_layer(self, repeat, in_channels, out_channels, stride, t):
        layers = []
        layers.append(BottleNeckt(in_channels, out_channels, stride, t))
        while repeat-1:
            layers.append(BottleNeckt(out_channels, out_channels, (1,1,1), t))
            repeat -= 1
        return nn.Sequential(*layers)

    def forward1(self, x):
        x = self.conv1(x)
        x1 = self.maxpool(x)
        x2 = self.tab1(self.bottleneckt1(x1))
        x3 = self.tab2(self.bottleneckt2(x2))
        x4 = self.tab3(self.bottleneckt3(x3))
        x5 = self.tab4(self.bottleneckt4(x4))
        x6 = self.globalavgpool(x5)
        out = x6.flatten(1)
        return x1,x2,x3,x4,out


if __name__ == '__main__':
    # test
    inputs = torch.rand(1, 3, 32, 224, 224)
    net = Temporalpath()

    _, _, _, _, outputs = net.forward(inputs)
    print('outputs:', outputs)
    print(outputs.size())
