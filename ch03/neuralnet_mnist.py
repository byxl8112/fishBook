import pickle
import sys, os
import numpy as np
from common.functions import sigmoid, softmax

sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
from dataset.mnist import load_mnist


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True,  # 把像素值从 0 - 255 压缩到 0-1
        flatten=True,  # 把 28 x 28 的图片展平为 784 个数字的一维数组
        one_hot_label=False)  # 标签用数字表示 0 - 9, ，不用 one-hot 编码
    return x_test, t_test


# 读入保存在pickle文件 sample_weight.pkl 中的学习到的权重参数（以字典变量的形式保存了权重和偏置参数）
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
        return network


# predict函数以Numpy数组的形式输出各个标签对应的概率
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()  # 获取测试图片和正确标签
network = init_network()   # 加载权重

accuracy_cnt = 0
# 用 for语句逐一取出保存在x中的图像数据，然后使用predict分类
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)  # 获取概率最高的元素的索引
    if p == t[i]:  # 预测结果和正确标签一致
        accuracy_cnt += 1  # 正确数量 +1

print("Accuracy: " + str(float(accuracy_cnt) / len(x)))
