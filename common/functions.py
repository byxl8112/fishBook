import numpy as np


# sigmoid：把任意数值压缩到 0 - 1 之间
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# softmax：把神经网络最后一层的输出变成各个类别的概率，概率最大的就是预测结果
def softmax(x):
    if x.ndim == 2:  # 判断是否是 二维数组
        x = x.T  # 转置，方便按列计算
        x = x - np.max(x, axis=0)  # 溢出对策
        y = np.exp(x) / np.sum(np.exp(x), axis=0)  # 计算 softmax
        return y.T  # 转回原来的形状
    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


# cross_entropy_error: 交叉熵损失函数实现，可同时处理单个样本和批量数据，也兼容 One-Hot 编码和 标签索引 两种格式标签
# return: 一个具体的数值（交叉熵损失值），代表了当前这批数值的平均损失，
# 它用来定量衡量模型的预测概率分布 y 与真实标签 t 之间的“差距”。损失值越接近 0，说明模型猜得越准。
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:  # 如果满足，则说明传入的是 one-hot 编码
        t = t.argmax(axis=1)  # 找出每一行中最大值（即数字1）所在的 索引
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
