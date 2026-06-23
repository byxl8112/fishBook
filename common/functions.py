import numpy as np


# sigmoid：把任意数值压缩到 0 - 1 之间
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# softmax：把神经网络最后一层的输出变成各个类别的概率，概率最大的就是预测结果
def softmax(x):
    if x.ndim == 2:    # 判断是否是 二维数组
        x = x.T    # 转置，方便按列计算
        x = x - np.max(x, axis=0)   # 溢出对策
        y = np.exp(x) / np.sum(np.exp(x), axis=0)     # 计算 softmax
        return y.T   # 转回原来的形状
    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

