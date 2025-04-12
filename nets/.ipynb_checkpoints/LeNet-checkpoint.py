from torch import nn
from torchinfo import summary

LeNet = nn.Sequential(
    nn.Conv2d(3, 6, kernel_size=5, padding=2), nn.ReLU(), # 224 x 224
    nn.AvgPool2d(kernel_size=2, stride=2), # 112 x 112
    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), # 108 x 108
    nn.AvgPool2d(kernel_size=2, stride=2),  # 54 x 54
    nn.Flatten(),
    nn.Linear(54 * 54, 120), nn.ReLU(),
    nn.Linear(120, 84), nn.ReLU(),
    nn.Linear(84, 56))

    
if __name__ == '__main__':
    
    print(summary(LeNet, (3, 224, 224)))



