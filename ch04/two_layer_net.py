import numpy as np
import sys, os

sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient


# 实现一个完整的、全功能两层神经网络（包含一个隐藏层和一个输出层）
class TwoLayerNet:
    # 初始化权重
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    # 计算预测值，实现标准的矩阵乘法和非线性转换
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']  # 隐藏层计算
        b1, b2 = self.params['b1'], self.params['b2']  # 输出层计算

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        y = softmax(z2)

        return y

    # 计算损失函数，x：输入数据，t：监督数据，return：当前错得多少的得分
    def loss(self, x, t):
        x_predict = self.predict(x)

        return cross_entropy_error(x_predict, t)

    # 计算准确率
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)  # 提取概率最大和真实标签中 1 所在的索引
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])  # 计算 这组数据的准确率有多少

        return accuracy

    # 定义求偏导，即计算W1, b1, W2, b2的梯度
    # 慢速求梯度，调用了 np.nditer 盲人摸象式数值求导。它让每个参数轮流加减 1e-4 跑一遍前向传播。如果有几万个参数，计算太慢了
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)  # 损失函数

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
