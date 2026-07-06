import numpy as np
import sys, os

sys.path.append(os.pardir)  # 为了导入父目录中的文件
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 初始化一个 2x3 的权重矩阵

    def predict(self, x):
        return np.dot(x, self.W)  # 计算输入x和权重W的距离乘法（点乘）

    def loss(self, x, t):
        z = self.predict(x)  # 得到神经网络的原始输出得分
        y = softmax(z)  # 通过 softmax 函数转化为概率分布（所有预测概率和为1）
        loss = cross_entropy_error(y, t)  # 计算预测概率和真实标签t之间的交叉熵误差（损失）

        return loss


x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()  # 初始化simpleNet类，可以使用里面的方法

f = lambda w: net.loss(x, t)  # 虽然参数是w，但实际不用这个参数
dW = numerical_gradient(f, net.W)  # 计算损失函数关于权重参数的梯度，参数 f 函数还是lambda这个函数，到numerical_gradient里面才解开

print(dW)
