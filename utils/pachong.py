import re
import requests
import time  # 时间模块
from urllib import parse  # 对汉字进行编码
import os  # 文件操作
from fake_useragent import UserAgent  # 随机生成一个user-agent


class Picture:

    def __init__(self):
        self.name_ = input('请输入关键字:')  # 输入要搜索的图片
        self.name = parse.quote(self.name_)  # URL只允许一部分ASCII字符，所以对搜索关键词进行编码
        self.times = str(int(time.time() * 1000))  # 返回当前的时间戳
        self.url = 'http://www.minzunet.cn/dcmz/ns/ctfz/1ba8870a-1.html'
        self.headers = {'User-Agent': UserAgent().random}

    # 请求30张图片的链接
    def get_one_html(self, url, pn):
        response = requests.get(url=url.format(self.name, self.name, pn, self.times),
                                headers=self.headers).content.decode('utf-8')
        return response

    # 请求单张图片内容  获取指定图片的UR
    def get_two_html(self, url):
        response = requests.get(url=url, headers=self.headers).content
        return response

    # 解析含30张图片的html的内容
    def parse_html(self, regex, html):
        content = regex.findall(html)
        return content

    # 主函数
    def run(self):
        # 判断该目录下是否存在与输入名称一样的文件夹 如果没有则创建 有就不执行if下的创建
        if not os.path.exists('./{}/'.format(self.name_)):  # 要创建一个文件保存图片，先看看这个文件名有没有重复
            os.mkdir('./{}'.format(self.name_))  # 如果存在就直接保存，如果不存在就用os.mkdir创建一个
        response = self.get_one_html(self.url, 1)
        regex1 = re.compile('"displayNum":(.*?),')
        num = self.parse_html(regex1, response)[0]  # 获取总的照片数量
        print('该关键字下一共有{}张照片'.format(num))  # 打印总的照片数量

        # 判断总数能不能整除1
        if int(num) % 1 == 0:
            pn = int(num) / 1
        else:
            # 总数量除30是因为每一个链接有30张照片 +2是因为要想range最多取到该数就需要+1
            # 另外的+1是因为该总数除30可能有余数，有余数就需要一个链接 所以要+1
            pn = int(num) // 1 + 2
        number = 0
        for i in range(pn):  # 遍历每一个含5张图片的链接
            resp = self.get_one_html(self.url, i * 1)
            regex2 = re.compile('"middleURL":"(.*?)"')
            urls = self.parse_html(regex2, resp)  # 得到30张图片的链接（30个）
            if number >= 100:
                break
            for u in urls:  # 遍历每张图片的链接
                try:
                    content = self.get_two_html(u)  # 请求每张图片的内容
                    number = number + 1  # 解除循环
                    # 打开该关键字下的文件写入图片
                    # 保存图片，名字是data的28-25，wb是二进制写入，f.write是保存图片
                    with open('./{}/{}.jpg'.format(self.name_, u[28:35]), 'wb') as f:
                        f.write(content)
                    print('完成一张照片')  # 汉族下载完一张图片后打印
                except requests.exceptions.MissingSchema:
                    pass


if __name__ == '__main__':
    pachong = Picture()
    pachong.run()
