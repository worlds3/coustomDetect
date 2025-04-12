import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchinfo import summary


class regnet_y_400mf(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.regnet_y_400mf(weights='IMAGENET1K_V1')
        self.model.fc.add_module('add_linear', torch.nn.Linear(1000, 56))
    def forward(self,x):
        x = self.model(x)
        return x

if __name__ == '__main__':

    net = regnet_y_400mf()
    print(summary(net, (16, 3, 224, 224)))
