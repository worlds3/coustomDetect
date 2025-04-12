import torch
from torch.utils.data import DataLoader


def getStat(train_data):
    print(len(train_data))  # 计算数据长度
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    # 从DataLoader里分批取数据，返回值为images, label
    mean = torch.zeros(3)
    # 初始化均值为(0,0,0)
    std = torch.zeros(3)
    # 初始化方差为(0,0,0)
    for X, label in train_dataloader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            # 计算每个batch数据的第d个通道的均值
            std[d] += X[:, d, :, :].std()
            # 计算每个batch数据的第d个通道的方差
    mean.div_(len(train_data))
    # 计算所有数据均值
    std.div_(len(train_data))
    # 计算所有数据方差
    return list(mean.numpy()), list(std.numpy())



