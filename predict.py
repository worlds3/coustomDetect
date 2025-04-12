import torch
from PIL import Image
from nets.nn_model import nets
from data_loader.transforms import get_test_transform

image = input('请输入图片路径：')  # 输入图片的绝对路径
img = Image.open(image)
img.show()
img = img.convert('RGB')  # png四通道,jpg三通道

# 进入网络的图片要进行预处理
trans = get_test_transform(size=300)
img = trans(img)
img = torch.unsqueeze(img, 0)

model = nets('resnet18')  # 调用模型
model.load_state_dict(torch.load('./models/10.pth', map_location=torch.device('cpu')))  # 将gpu训练的模型在cpu上跑
print('------模型已加载完毕-------')
classes = ['achang', 'bai', 'baoan', 'bulang', 'buyi', 'chaoxain', 'dai', 'dawoer', 'deang',
           'dong', 'dongxiang', 'dulong', 'elunchun', 'eluosi', 'ewenke', 'gaoshan', 'gelao', 'hani',
           'han', 'hasake', 'hezhe', 'hui', 'jing', 'jingpo', 'jinuo', 'keerkezi', 'lahu',
           'li', 'luoba', 'man', 'maonan', 'menba', 'menggu', 'miao', 'mulao', 'naxi', 'nu',
           'pumi', 'qiang', 'sala', 'she', 'shui', 'susu', 'tajike', 'tataer', 'tujia', 'tu',
           'wa', 'weiwuer', 'wuzibieke', 'xibo', 'yao', 'yi', 'yugu', 'zang', 'zhaung'
           ]
model.eval()
with torch.no_grad():
    output = model(img)
    class_pre = output.argmax(1).item()  # 预测类
    print(f'预测结果为：{classes[class_pre]} zu')


