# Code for "An End-to-End Two-Branch Network Towards Robust Video Fingerprinting"
# Yingying Xu, Yuanding Zhou, Xinran Li, Gejian Zhao, and Chuan Qin

import torch
import torch.nn as nn
from model.temporal import Temporalpath
from torchsummary import summary

class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t, second=False):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, t, (1,1,1), stride=(1,1,1), padding=(0,0,0), bias=False),
            nn.BatchNorm3d(t),
            nn.ReLU(inplace=True),
        )

        if second:
            self.conv1 = nn.Sequential(
                nn.Conv3d(t, t, (1,3,3), stride=(1,1,1), padding=(0,2,2), groups=t,
                          dilation=(1,2,2), bias=False),
                nn.BatchNorm3d(t),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv3d(t, t, (1, 3, 3), stride=stride, padding=(0, 1, 1),
                          groups=t, bias=False),
                nn.BatchNorm3d(t),
                nn.ReLU(inplace=True),
            )

        if stride == (1, 1, 1):
            self.conv2 = nn.Sequential(
                nn.Conv3d(t, out_channels,  (1,1,1), stride=(1,1,1), padding=(0,0,0), bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv3d(t + in_channels, out_channels,  (1,1,1), stride=(1,1,1), padding=(0,0,0), bias=False),
                nn.BatchNorm3d(out_channels)
            )

        self.shortcut = nn.Sequential()
        if stride == (1,1,1) and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, (1,1,1), stride=(1,1,1), padding=(0,0,0), bias=False),
                nn.BatchNorm3d(out_channels)
            )

        self.stride = stride
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.conv1(x1)
        if self.stride != (1,1,1):
            x0 = self.maxpool(x)
            x2 = torch.cat((x0,x2), 1)

        out = self.conv2(x2)

        if self.stride == (1,1,1):
            out += self.shortcut(x)

        return out


class Spatialpath(nn.Module):
    def __init__(self,hash_bits=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 32, (1,7,7), stride=(1,2,2), padding=(0,3,3), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.bottleneck1 = self.make_layer(2, 32, 64, (1,1,1), 128)
        self.bottleneck2 = self.make_layer(2, 64, 128, (1,2,2), 256)
        self.bottleneck3 = self.make_layer(2, 128, 256, (1,2,2), 512)
        self.bottleneck4 = self.make_layer(2, 256, 512, (1,2,2), 1024)
        self.globalavgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.fc = nn.Linear(768, hash_bits)
        self.tanh = nn.Tanh()

        self.temporal = Temporalpath()
        self.hash_bits = hash_bits

        Ks = [32, 64, 128]

        self.convsf = nn.Sequential(
            nn.Conv3d(8, 32, (7, 1, 1), stride=(4, 1, 1), padding=(3, 0, 0), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        self.convsf1 = nn.ModuleList([nn.Conv3d(K, K*2, (7,1,1), stride=(4,1,1),
                                              padding=(3,0,0), bias=False) for K in Ks])

        self.bnf = nn.BatchNorm3d(64)
        self.bnf1 = nn.BatchNorm3d(128)
        self.bnf2 = nn.BatchNorm3d(256)

        self.reluf = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def make_layer(self, repeat, in_channels, out_channels, stride, t):
        layers = []
        layers.append(BottleNeck(in_channels, out_channels, stride, t))
        while repeat-1:
            layers.append(BottleNeck(out_channels, out_channels, (1,1,1), t, second=True))
            repeat -= 1
        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in')
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x1, x2, x3, x4, out = self.temporal.forward1(x)

        x1 = self.convsf(x1)
        x2 = self.reluf(self.bnf(self.convsf1[0](x2)))
        x3 = self.reluf(self.bnf1(self.convsf1[1](x3)))
        x4 = self.reluf(self.bnf2(self.convsf1[2](x4)))

        B,C,T,H,W = x.size()

        x = x[:, :, 0:T:4, :, :]

        # spatial branch
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bottleneck1(x+x1)
        x = self.bottleneck2(x+x2)
        x = self.bottleneck3(x+x3)
        x = self.bottleneck4(x+x4)
        x = self.globalavgpool(x)
        x = x.flatten(1)

        final = torch.cat((x,out),1)
        final = self.fc(final)

        return out, x, final

if __name__ == '__main__':
    # test
    # inputs = torch.rand(1, 3, 32, 224, 224)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Spatialpath().to(device)

    summary(net, (3, 32, 224, 224))