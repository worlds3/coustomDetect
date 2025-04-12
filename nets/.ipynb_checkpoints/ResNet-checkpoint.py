import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchinfo import summary

"""
从0搭建的resnet18
"""

# 定义残差块
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        # 第一次卷积
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        # 第二次卷积
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        # 定义残差加的模块
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        # 两个批量归一化层
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        # 完成第一次卷积，第一次批量归一化和第一次relu操作
        Y = F.relu(self.bn1(self.conv1(X)))
        # 完成第2次卷积，第2次批量归一化
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        # 计算f(x)+x
        Y += X
        return F.relu(Y)

"""
ResNet的前两层跟之前介绍的GoogLeNet中的一样： 在输出通道数为64、步幅为2的卷积层后，
接步幅为2的的最大池化层。 不同之处在于ResNet每个卷积层后增加了批量规范化层。
"""
b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

resnet18_pre_none = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(), nn.Linear(512, 56))


class resnet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights='IMAGENET1K_V1')
        self.model.fc = nn.Linear(self.model.fc.in_features, 56)
            
    def forward(self,x):
        x = self.model(x)
        return x


class resnet101(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet101(pretrained=False)
        self.model.load_state_dict(torch.load('/root/autodl-tmp/tradition_cloth_classify/weights/resnet101-5d3b4d8f.pth'))
        self.model.fc = nn.Linear(self.model.fc.in_features, 56)
    def forward(self,x):
        x = self.model(x)
        return x


if __name__ == '__main__':

    net = resnet101()
    print(summary(net, (16, 3, 224, 224)))
