import numpy as np
import matplotlib.pyplot as plt

# 目标函数：y = x^2
def square_func(x):
    return np.square(x)

# 目标函数的一阶导数 dy/dx = 2x
def dfunc(x):
    return 2 * x


# 梯度下降
def gd_decay(x_start,df,epochs,lr,decay):
    """
    :param x_start:初始位置
    :param df:平方函数的导数函数
    :param epochs:迭代次数
    :param lr:学习率
    :param decay:学习率衰减系数
    :return:
    """

    xs = np.zeros(epochs + 1)
    x = x_start
    xs[0] = x
    y = 0
    # 循环：迭代计算下一次x的位置

    for i in range(epochs):
        dx = df(x)
        #学习率衰减
        lr_i = lr * 1.0 /(1.0 + decay * i)
        y = -dx * lr_i
        x += y
        xs[i+1] = x
    return xs


if __name__ == "__main__":
    # 生成基础采样点给后面使用（以及画图）
    line_x = np.linspace(-5, 5, 100)
    line_y = square_func(line_x)

    x_start = 5
    epochs = 10

    lr = [0.1, 0.3, 0.9, 0.09]
    decay = [0.0, 0.01, 0.5, 0.9]

    color = ['k', 'r', 'g', 'y']

    # 双重验证并绘制图像
    plt.figure("Gradinet Descent：Decay", figsize=(14, 10))
    row = len(lr)
    col = len(decay)
    size = np.ones(epochs + 1) * 10
    size[-1] = 70
    for i in range(row):
        for j in range(col):
            x = gd_decay(x_start, dfunc, epochs, lr=lr[i], decay=decay[j])
            plt.subplot(row, col, i * col + j + 1)
            plt.plot(line_x, line_y, c='b')
            plt.plot(x, square_func(x), c=color[i], label='lr={},de={}'.format(lr[i], decay[j]))
            plt.scatter(x, square_func(x), c=color[i], s=size)
            plt.legend(loc=0)
    plt.show()


