import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchinfo import summary

class densenet169(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.densenet169(weights='IMAGENET1K_V1')
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 56)
            
    def forward(self,x):
        x = self.model(x)
        return x

if __name__ == '__main__':

    net = densenet169()
    print(summary(net, (16, 3, 224, 224)))
