import torchvision
import torch
from nets.ResNet import resnet18_pre_none, resnet50, resnet101
from nets.LeNet import LeNet
from nets.AlexNet import AlexNet
from nets.convnet import convnext_small
from nets.DenseNet import densenet169
from nets.efficient import eff_b1

def nets(type):
    if type == 'resnet18':
        print(f'使用resnet18进行训练：')
        resnet18 = torchvision.models.resnet18(progress=True)
        resnet18.load_state_dict(torch.load('/root/autodl-tmp/tradition_cloth_classify/weights/resnet18-5c106cde.pth'))
        resnet18.fc.add_module('add_linear', torch.nn.Linear(1000, 56))
        # print(*[(name, param.shape) for name, param in resnet18.named_parameters()])
        return resnet18
    elif type == 'lenet':
        print(f'使用lenet进行训练：')
        return LeNet
    elif type == 'resnet18_pre_none': # 没有预训练的。
        print(f'使用resnet18_pre_none进行训练：')
        return resnet18_pre_none
    elif type == 'alexnet':
        print(f'使用alexnet进行训练：')
        return AlexNet
    elif type == 'resnet50':
        print(f'使用resnet50进行训练：')
        return resnet50()
    elif type == 'densenet169':
        print(f'使用densenet169进行训练：')
        return densenet169()
    elif type == 'convnext_small':
        print(f'使用convnext_small进行训练：')
        return convnext_small()
    elif type == 'eff_b1':
        print("使用eff_b1进行训练：")
        return eff_b1()
    elif type == 'resnet101':
        print(f'使用resnet101进行训练：')
        return resnet101()
    else:
        print(f'输入错误！')


# if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = nets('resnet101')
    # print(net)





