import torch
from torch import nn
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from data_loader.Dataset import train_dataloader, test_dataloader, trainloader_size, testloader_size
from nets.nn_model import nets
from sklearn.metrics import f1_score
from logger.logger import setup_logger


def train(model, name, train_dataloader, test_dataloader, criterion, optimizer, scheduler, exp='exp1', epochs=100):
    start = datetime.now()
    best_test_acc = 0
    best_weights = None
    best_log = None
    # tensorboard
    writer = SummaryWriter(f'./logs/{exp}/{name}')

    # 日志
    logger = setup_logger(name, exp)
    logger.info(f"Training started for model: {name}")

    for epoch in range(epochs):
        print(f'-----------------第{epoch+1}轮训练-----------------')
        start_epoch = datetime.now()
        epoch_loss = 0 # 每轮的损失
        epoch_corrects = 0 # 每轮的正确数

        # 存放预测和真实labels，计算f1
        all_train_preds = []  
        all_train_labels = []

        model.train()  
        for batch, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device) # 移动到设备上
            outputs = model(images) #  前向传播
            loss = criterion(outputs, labels.long())  # 要求输入为long，计算loss
            optimizer.zero_grad() # 每个batch训练，梯度要清零
            loss.backward() # 反向传播
            optimizer.step() # 梯度更新

            _, preds = torch.max(outputs, 1) # 预测的最大值
            epoch_loss += loss.item() 
            epoch_corrects += torch.sum(preds == labels.data) # 这个batch预测正确的数量

            # 保存这个batch算f1分数
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

            if batch % 50 == 0:
                print('第{}个batch, train Loss:{}'.format(batch, loss.item()))  
            
            
        end_epoch = datetime.now()
        epoch_loss /= trainloader_size # 这一轮损失
        epoch_acc = epoch_corrects.double() / len(train_dataloader.dataset) # 这一轮acc
        epoch_f1 = f1_score(all_train_labels, all_train_preds, average='macro') # 这一轮f1

        # 训练的日志
        train_log = f'Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Train F1: {epoch_f1:.4f}'

        print(f'本epoch耗时{end_epoch - start_epoch}')
        writer.add_scalar('loss/train', epoch_loss, epoch)
        writer.add_scalar('acc/train', epoch_acc, epoch)
        writer.add_scalar('f1/train', epoch_f1, epoch)


    # 每个epoch测试一次损失
        model.eval()  
        test_loss = 0 # 测试损失
        test_corrects = 0 # 测试正确数
        
        # 保存算f1
        all_test_labels = []
        all_test_preds = []

        with torch.no_grad():
            for batch, (images, labels) in enumerate(test_dataloader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images) # 前向
                loss = criterion(outputs, labels.long())  # 要求输入为long，计算loss

                _, preds = torch.max(outputs, 1) 
                test_loss += loss.item()
                test_corrects += torch.sum(preds == labels.data)
                all_test_labels.extend(labels.cpu().numpy())
                all_test_preds.extend(preds.cpu().numpy())

        test_loss /= testloader_size
        test_acc = test_corrects.double() / len(test_dataloader.dataset)
        test_f1 = f1_score(all_test_labels, all_test_preds, average='macro')

        test_log = f' Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}'
        epoch_log = train_log + test_log # 这一轮的日志
        logger.info(epoch_log)
        print(epoch_log)

        writer.add_scalar('loss/test', test_loss, epoch)
        writer.add_scalar('acc/test', test_acc, epoch)
        writer.add_scalar('f1/test', test_f1, epoch)

        # 保存模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_log = "最好的一次：" + epoch_log
            best_weights = model.state_dict()

        
    if best_weights:
        # 模型的权重
        if not os.path.exists(f"./weights/{exp}/{name}"):
            os.makedirs(f"./weights/{exp}/{name}")
        torch.save(best_weights, './weights/{}/{}/{}_{}.pth'.format(exp, name, name, int(best_test_acc*10000)))

    writer.close()
    end = datetime.now()
    end_log = f"Training ended.\nTotal time: {end - start}"
    logger.info(end_log)
    logger.info(best_log)
    print(end_log)


if __name__ == '__main__':

    # 超参
    lr, epochs, weight_decay = 1e-4, 30, 1e-4
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    # 训练
    # model_list = ['resnet18_pre_none', 'resnet18',  'resnet50', 'densenet169', 'convnext_small', 'eff_b1']
    # model_list = ['eff_b1', 'resnet18']
    model_list = ['resnet101']
    
    for name in model_list:
        # 获取model
        model = nets(name).to(device)
        # 优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        # 练！
        train(model, name, train_dataloader, test_dataloader, criterion, optimizer, scheduler=None, exp='exp3_aug_v5', epochs=epochs)
