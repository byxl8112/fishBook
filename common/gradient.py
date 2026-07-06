import numpy as np


# 用于处理一维数组
def numerical_gradient_1d(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)

        grad[idx] = (fxh1 + fxh2) / (2 * h)  # 求导

        x[idx] = tmp_val  # 还原x的值

    return grad


# 用于处理二维数组矩阵
def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):  # 提取出二维数组X的索引idx和里面的一维数组x
            grad[idx] = numerical_gradient_1d(f, x)

    return grad


# 用于处理多维数组
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])  # multi_index表示开启多维度索引，readwrite表示迭代过程中可以改写x中的值
    while not it.finished:
        idx = it.multi_index  # 拿到当前元素的 “多维坐标元组”
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)  # 求偏导

        x[idx] = tmp_val  # 还原值
        it.iternext()  # 指针移动到下一个元素

    return grad
















