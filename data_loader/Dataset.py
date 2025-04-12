import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from .transforms import *

# 输入大小和batch_size大小
input_size = 300
batch_size = 128

class SelfCustomDataset(Dataset):
    def __init__(self, label_file, imageset):
        super(SelfCustomDataset, self).__init__()  # 继承torch中的Dataset类
        self.img_aug = True
        with open(label_file, 'r') as f:
            # 因为给图片命名标签的时候，是图片名+空格+标签存放在txt文件中的，所以这里要按空格分隔开input和label
            self.imgs = list(map(lambda line: line.strip().split(' '), f))
            # 打开图片文件夹，去除换行符，以空格作为分隔生成列表
            # [['../images/test\\achang\\1.jpg', '0'], ... ]
        if imageset == 'train':
            self.transform = get_train_transform_v4(size=input_size)
        else:
            self.transform = get_test_transform(size=input_size)
        self.input_size = input_size

    def __getitem__(self, index):
        # 按照上面分过后的imgs就是一个包含图片路径名，标签的列表。
        img_path, label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        # 保持图片格式为三通道RGB格式

        if self.img_aug:  # 做数据增强
            img = self.transform(img)
        else:
            # 不做数据增强就先从pillow转换为numpy，然后转换为tensor
            img = np.array(img)
            img = torch.from_numpy(img)

        return img, torch.from_numpy(np.array(int(label))) # 对标签先取整，接着转化为numpy，最后转为tensor

    def __len__(self):
        return len(self.imgs)

# 训练集加载为loader
train_label_dir = '/root/autodl-tmp/tradition_cloth_classify/images/train.txt'
train_datasets = SelfCustomDataset(train_label_dir, imageset='train')
train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, drop_last=True)
trainloader_size = len(train_dataloader) # loader中的batch数

# 测试集加载为loader
test_label_dir = '/root/autodl-tmp/tradition_cloth_classify/images/test.txt'
test_datasets = SelfCustomDataset(test_label_dir, imageset='test')
test_dataloader = DataLoader(test_datasets, batch_size=batch_size, shuffle=True, drop_last=True)
testloader_size = len(test_dataloader) # loader中的batch数


if __name__ == "__main__":
    
    # 打印一下数量信息
    print("trainloader的batch数:", trainloader_size)
    print("testloader的batch数:", testloader_size)
    print("traindataset的image数:", len(train_datasets))
    print("testdataset的image数:", len(test_datasets))

    # images, labels形状
    loader_iter = iter(test_dataloader)
    images, labels = next(loader_iter)
    print("images:{}, labels:{}".format(images.shape, labels.shape))

    