import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 relu=True,
                 bn=True):
        super(BasicConv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.bn = nn.BatchNorm2d(in_channels) if bn else None
        self.relu = nn.ReLU() if relu else None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding)

    def forward(self, x):
        if self.bn:
            x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        x = self.conv(x)
        return x


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv1_1 = BasicConv(3, 8, kernel_size=3, stride=2, relu=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = BasicConv(8, 16, kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1 = BasicConv(16, 32, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        return x


class Net(nn.Module):
    def __init__(self, class_quantity=3):
        super(Net, self).__init__()
        self.class_quantity = class_quantity
        self.backbone = Backbone()
        self.prediction_conv = nn.Conv2d(32,
                                         2 + class_quantity,
                                         kernel_size=3,
                                         padding=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.prediction_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size(0), -1, x.size(-1))
        return x
