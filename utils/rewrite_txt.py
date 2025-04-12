import os
import glob
import random


if __name__ == '__main__':
    train_data_path = '/root/autodl-tmp/tradition_cloth_classify/images/train'
    test_data_path = '/root/autodl-tmp/tradition_cloth_classify/images/test'
    # 还是民族的列表
    train_labels = os.listdir(train_data_path)
    test_labels = os.listdir(test_data_path)


    # 56个民族的列表，一个一个取出来，带索引。
    # 也就是按字典序分配标签了。
    for index, label in enumerate(train_labels):

        # 直接就找到某个民族下面的图片了：比如images/train/hanzu/00.jpg，glob是匹配。
        train_img_list = glob.glob(os.path.join(train_data_path, label, '*.jpg'))
        # 打乱图片顺序
    
        random.shuffle(train_img_list)
        # 划分测试集与训练集
        # train_list = train_img_list[:int(0.8*len(img_list))]
        # testlist = train_img_list[(int(0.8*len(img_list)):]
        with open('../images/train.txt', 'a')as f:
            for img in train_img_list:
                # 图片名+民族索引（也就类别名）
                f.write(img + ' ' + str(index))
                f.write('\n')

    # 写test.txt文件
    for index, label in enumerate(test_labels):
        test_img_list = glob.glob(os.path.join(test_data_path, label, '*.jpg'))
        random.shuffle(test_img_list)
        with open('../images/test.txt', 'a')as f:
            for img in test_img_list:
                f.write(img + ' ' + str(index))
                f.write('\n')


