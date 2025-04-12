import pandas as pd

input_csv = "/root/autodl-tmp/tradition_cloth_classify/images/train_addition/valid_classes.csv"  # 需要添加的图片
output_txt = "/root/autodl-tmp/tradition_cloth_classify/images/train.txt"  # 加到图片路径中去
image_root_path = "/root/autodl-tmp/tradition_cloth_classify/images/train_addition/valid/" # 图片前要加路径

data = pd.read_csv(input_csv)

# 打开train.txt，追加模式
with open(output_txt, 'a') as f:
    # 遍历csv
    for index, row in data.iterrows():
        filename = row['filename']
        # 遍历所有的标签列，找到值为1的那个。
        for label in row.index[1:]:
            if row[label] == 1:  # 这张图片真实的标签
                # 写进去，其那面加一个路径，然后是图片名和标签
                f.write(image_root_path + f"{filename} {label.strip()}\n")
            

print(f"Data has been appended to {output_txt}.")