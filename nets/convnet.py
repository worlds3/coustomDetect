import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchinfo import summary

class convnext_small(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.convnext_small(weights='IMAGENET1K_V1')
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, 56)
            
    def forward(self, x):
        x = self.model(x)
        return x
    
if __name__ == '__main__':

    net = convnext_small()
    print(summary(net, (16, 3, 224, 224)))
