# 添加日志记录
import logging

def setup_logger(name, exp):
    
    # 创建logger，设置级别
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 创建文件handler
    handler = logging.FileHandler(f'./logs/{exp}/{name}/{name}.log', mode='a')
    handler.setLevel(logging.INFO)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)

    # 添加handler到logger
    logger.addHandler(handler)
    return logger