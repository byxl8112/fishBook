import numpy as np
import matplotlib.pylab as plt
import matplotlib

matplotlib.use("TkAgg")


# 处理一维的值（即非批量计算）  x = [1,2]
# return: 斜率
def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # h = 0.0001，一个小值，使用定义计算导数
    grad = np.zeros_like(x)  # 复制一个x数组，且值赋值为0

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # 若 idx=0 ，则相当于 f(x0+h)，而x1不变

        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值

    return grad


# 多维，批量计算，X  = [[1,2], [3,4], ...]
# return: 斜率
def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

    return grad


# 平方求和
def function_2(x):
    if x.ndim == 1:
        return np.sum(x ** 2)
    else:
        return np.sum(x ** 2, axis=1)  # 多维数组，以一行为单位进行相加


# 求切线方程
# return: 切线上的值（可以是数组）
def tangent_line(f, x):
    d = numerical_gradient(f, x)  # 求导数、切线
    b = f(x) - d * x  # y轴截距
    return lambda t: d * t + b


if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)  # 把两个一维坐标数组扩展成二维坐标网格（Grid），从而得到平面上所有坐标点

    # 使两个数组的位置一一对应
    X = X.flatten()  # 将一个二维以上的数组按行展开成一个一维数组
    Y = Y.flatten()

    grad = numerical_gradient(function_2, np.array([X, Y]))  # 计算函数function_2在所有坐标点上的梯度（分为 偏x0求导和 偏x1求导），共两个数组

    plt.figure()  # 创建一个新的窗口
    plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")  # 画箭头，在(x,y)位置画一个方向向量为(-grad[0], -grad[1])的箭头
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()  # 显示网格线，默认开启 = plt.grid(True)
    # plt.legend()  # 显示图例，告诉哪条线代表什么
    plt.draw()  # 立即刷新当前图像
    plt.show()
