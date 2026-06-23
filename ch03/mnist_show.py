import sys, os
import numpy as np

sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
from dataset.mnist import load_mnist
from PIL import Image  # 图像显示


# (训练图像，训练标签)，(测试图像，测试标签)
# normalize ：是否将输入图像正规化为 0.0 - 1.0 的值，如果为False，则输入图像的像素会保持原来的 0 - 255.
# flatten ：是否展开输入图像（变成一维数组）。如果为False，则输入图像为 1 x 28 x 28 的三维数组，如果为True，则输入图像会保存为784个元素构成的一维数组
# one_hot_label：是否将标签保存为 one-hot表示（one-hot representation）。one-hot表示是仅正确解标签为1，其余全为0的数组，如 [0, 0, 1]
# 当 one_hot_label 为False时，只是像 7 , 2 这样简单保存正确解标签，当 one_hot_label 为True时，标签则保存为 one-hot表示

# (x_train, t_label), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# 输出各个数据的形状
# print(x_train.shape)
# print(t_label.shape)
# print(x_test.shape)
# print(t_test.shape)


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
img = x_train[2]
label = t_train[2]
print(label)

# img 原来的尺寸和重塑的尺寸
print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)


















