import sys, os

sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import numpy as np
import matplotlib.pylab as plt
import matplotlib

matplotlib.use("TkAgg")

# normalize=True：把图片中0-255的像素值缩放到0.0-1.0之间（归一化，让网络训练更稳定）
# x 表示input输入数据/特征，t 表示真实标签/正确答案。train表示训练集/练习题(基本不变)，test表示测试集/高考试卷(一直变)
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)  # 初始化数据数组，网络尺寸对齐

iters_num = 10000  # 总共循环迭代 10000 步
train_size = x_train.shape[0]  # 训练数据总数有多少
batch_size = 100  # 每次给网络 100 张图片来训练，即 Mini-batch
learning_rate = 0.1  # 学习率

train_loss_list = []  # 保存 loss，以后画 loss 曲线
train_acc_list = []  # 保存准确率
test_acc_list = []  # 测试准确率

# 算出多少步为一个 epoch（意味着模型每走 600 步，相当于把 60000张图片轮流看了一遍），看完整套教材一次，就叫一个 epoch
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)  # train_size 范围内选出来 batch_size 个数据
    x_batch = x_train[batch_mask]  # 将选出来的图片数据打包拿出来，变成 100x784 的矩阵
    t_batch = t_train[batch_mask]  # 将选出来的图片对应的标签拿出来，变成 100x10 的矩阵

    # 计算梯度
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)  # 反向传播计算梯度

    # 更新参数
    for key in ('W1', 'W2', 'b1', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)  # 求损失函数
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:  # 每经过一次 epoch 记录一次数据（准确率）
        train_acc = network.accuracy(x_train, t_train)  # 计算模型在训练集上的准确率，看看模型是不是死记硬背的能力，如果很低说明模型太笨了
        test_acc = network.accuracy(x_test, t_test)  # 计算模型在测试集上的准确率，测试集图片没有参与过计算和更新，看有没有泛化能力
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("训练数据，测试数据 | " + str(train_acc) + "," + str(test_acc))

# 画图
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))

plt.plot(x, train_acc_list, label='train_acc')
plt.plot(x, test_acc_list, label='test_acc')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

"""
总结：
整个训练过程中，输入数据（x_train、t_train、x_test、t_test）保持不变，模型之所以不断进步，
是因为 network.params（即 W1、b1、W2、b2）不断更新。
grad 本身并不会累积变化，它只是每一步根据当前参数重新计算出来的梯度。
"""