import numpy as np
import matplotlib.pylab as plt
import matplotlib

matplotlib.use("TkAgg")


# return: 函数f求导后的值
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


# 定义切线函数
def tangent_line(f, x):
    d = numerical_diff(f, x)  # 计算 f(x) 在 x 点处的导数（切线斜率）
    b = f(x) - d * x  # 已知点(x, f(x))，切线为：f(x) = d*x + b ，求 切线的 截距b
    return lambda t: d * t + b  # 返回切线方程函数所有点的值


# f(x) 的图像
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)  # 返回切线方程函数所有点的值

plt.plot(x, y)  # f(x) 函数
plt.plot(x, y2)  # 在 x=5 处的切线
plt.show()
