import sys, os

sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

# one-hot 表示（正确解为1， 其余为0）
(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True,
    flatten=True,
    one_hot_label=True)


# print(x_train.shape)   # (60000, 784)
# print(t_train.shape)   # (60000, 10)

# train_size = x_train.shape[0]  # 总共 train_size 个训练数据
# batch_size = 10
# batch_mask = np.random.choice(train_size, batch_size)  # 从 train_size 个训练数据中随机选出 batch_size 个训练数据


# 批量数据函数，使用one-hot形式
def cross_entropy_error_oh(y, t):
    if y.ndim == 1:  # 一维
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


# 监督数据不是 one-hot 形式
def cross_entropy_error_not_oh(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
