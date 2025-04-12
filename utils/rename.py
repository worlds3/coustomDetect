import os


def rename():
    train_path = '../images/train'
    test_path = '../images/test'
    # 这里是民族的列表
    train_lists = os.listdir(train_path)
    test_lists = os.listdir(test_path)
    fileType = '.jpg'

    # 修改训练集
    train_count = 0
    train_startNumber = 1

    # 具体取到哪个民族目录下
    for train_list in train_lists:
        one_class_path = os.path.join(train_path, train_list) # 比如现在是汉族目录
        one_class_list = os.listdir(one_class_path) # 汉族目录下的文件列表
        for file in one_class_list: # 取图片文件
            print("正在生成以" + str(one_class_path) + str(train_count) + fileType + "迭代的文件名") # 生成以images/train/hanzu + 个数 + .jpg 的文件
            Old_dir = os.path.join(one_class_path, file)  # 旧文件名：images/train/hanzu/889797.jpg
            # 如果是目录就跳过，逻辑有错。
            if os.path.isdir(Old_dir):  
                continue
            # images/train/hanzu + 0.jpg
            New_dir = os.path.join(one_class_path, str(train_count+int(train_startNumber))+fileType)
            os.rename(Old_dir, New_dir)
            train_count += 1
    print("训练集一共修改了"+str(train_count)+"个文件")

    # 修改测试集
    test_count = 0
    test_startNumber = 1
    for test_list in test_lists:
        one_class_path = os.path.join(test_path, test_list)
        one_class_list = os.listdir(one_class_path)
        for file in one_class_list:
            print("正在生成以" + str(one_class_path) + str(test_count) + fileType + "迭代的文件名")
            Old_dir = os.path.join(one_class_path, file) 
            if os.path.isdir(Old_dir):  
                continue
            New_dir = os.path.join(one_class_path, str(test_count+int(test_startNumber))+fileType)
            os.rename(Old_dir, New_dir)
            test_count += 1
    print("测试集一共修改了"+str(test_count)+"个文件")
    

if __name__ == '__main__':
    rename()
