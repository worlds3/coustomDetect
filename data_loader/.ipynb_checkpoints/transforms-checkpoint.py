from torchvision import transforms
import numpy as np
import torch
from PIL import Image

############################################# baseline 数据增强 ####################################

def get_train_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), size=0):
    return transforms.Compose([
        transforms.Resize((300, 300)),  # 图片尺寸归一化
        transforms.ToTensor(),  # PIL格式转换为tensor，在神经网络中训练
        transforms.Normalize(mean=mean, std=std),
        # 使用Imagenet的均值和标准差，将3个通道的数据进行归一化
    ])


def get_test_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), size=0):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


############################################# v1 数据增强 ####################################

def get_train_transform_v1(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), size=0):
    return transforms.Compose([
        transforms.Resize(size=(224, 224)), 
        transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
        transforms.RandomVerticalFlip(p=0.5),  # 垂直翻转
        transforms.RandomRotation(degrees=15, expand=False, center=None),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # 颜色抖动
        transforms.ToTensor(),  # PIL格式转换为tensor，在神经网络中训练
        transforms.Normalize(mean=mean, std=std),
        # 使用Imagenet的均值和标准差，将3个通道的数据进行归一化
    ])

############################################# v2 数据增强 ####################################

# 高斯椒盐噪声
def add_gaussian_noise(image, mean=0, std=0.01):
    np_image = np.array(image) / 255.0  # 归一化到 [0, 1]
    noise = np.random.normal(mean, std, np_image.shape)
    np_image = np.clip(np_image + noise, 0, 1) * 255  # 加噪声并归一化回 [0, 255]
    return Image.fromarray(np_image.astype(np.uint8))

class AddGaussianNoise:
    def __init__(self, mean=0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        return add_gaussian_noise(img, self.mean, self.std)
    
# 加噪声
def get_train_transform_v2(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), size=0):
    return transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  
        transforms.RandomVerticalFlip(p=0.5),  # 垂直翻转
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),  # 仿射变换
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # 颜色抖动
        AddGaussianNoise(mean=0, std=0.01), # 椒盐噪声
        transforms.ToTensor(),  # PIL格式转换为tensor，在神经网络中训练
        transforms.Normalize(mean=mean, std=std),
        # 使用Imagenet的均值和标准差，将3个通道的数据进行归一化
    ])

############################################# v3 数据增强 ####################################

# 源于--bxd1
img_size=260
crop_size=20

class Cutout(object):
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length
    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img


def get_train_transform_v3(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), size=0):
    return transforms.Compose([
        transforms.Resize((img_size+crop_size,img_size)),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),  
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15, expand=False, center=None),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # 颜色抖动
        transforms.ToTensor(),  # PIL格式转换为tensor，在神经网络中训练
        Cutout(),
        transforms.RandomErasing(0.3),
        transforms.Normalize(mean=mean, std=std),
        # 使用Imagenet的均值和标准差，将3个通道的数据进行归一化
    ])

############################################# v4 数据增强 ####################################
def get_train_transform_v4(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), size=0):
    return transforms.Compose([
        transforms.Resize(size=(224, 224)), 
        transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
        transforms.RandomVerticalFlip(p=0.5),  # 垂直翻转
        transforms.RandomRotation(degrees=15, expand=False, center=None),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),  # 仿射变换
        transforms.ToTensor(),  # PIL格式转换为tensor，在神经网络中训练
        transforms.Normalize(mean=mean, std=std),
        # 使用Imagenet的均值和标准差，将3个通道的数据进行归一化
    ])