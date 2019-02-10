import tensorflow as tf
# 这两行代码设置日志输出级别，可以屏蔽TF没有采用源码安装无法加速的日志输出
import os
os.environ['TF_CPP_MIN_LOOG-LEVEL']='2'

def linear_regression():
    """
    TF实现线性回归
    :return:
    """
    # 准备真实数据：100个样本
    x = tf.random.normal(shape=[100,1])
    y_true = tf.matmul(x,[[0.8]]) + 0.7

    # 构建模型
    #　用变量定义模型参数
    weights = tf.Variable(initial_value=tf.random.normal(shape=[1,1]))
    bias = tf.Variable(initial_value=tf.random.normal(shape=[1,1]))
    y_predict = tf.matmul(x,weights) + bias

    # 构造损失函数(均方误差)
    loss = tf.reduce_mean(tf.square(y_predict-y_true))

    # 优化损失(梯度下降法)
    optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    # 显式初始化变量
    init = tf.global_variables_initializer()

    # 开启会话
    with tf.Session() as sess:
        # 先初始化变量
        sess.run(init)
        # 训练前的模型参数
        print("训练前的模型参数,权重：%f,偏置：%f,损失：%f" %(weights.eval(),bias.eval(),loss.eval()))

        # 开始训练
        for i in range(1000):
            sess.run(optimizer)
            print("第%d次训练后的模型参数,权重：%f,偏置：%f,损失：%f" % (i+1,weights.eval(), bias.eval(), loss.eval()))

    return None

def linear_regression2():
    """
    TF实现线性回归
    增加其它功能
    1、变量TensorBoard显示
    2、增加命名空间
    3、模型保存与加载
    4、命令行参数设置
    :return:
    """
    # 增加命名空间和指令名称，使用代码结构更清晰，TensorBoard可视化图结构更清楚
    with tf.variable_scope("prepare_data"):
        # 准备真实数据：100个样本
        x = tf.random.normal(shape=[100,1],name="features")
        y_true = tf.matmul(x,[[0.8]]) + 0.7

    with tf.variable_scope("create_model"):
        # 构建模型
        #　用变量定义模型参数
        weights = tf.Variable(initial_value=tf.random.normal(shape=[1,1]),name="Weights")
        bias = tf.Variable(initial_value=tf.random.normal(shape=[1,1]),name="Bias")
        y_predict = tf.matmul(x,weights) + bias

    with tf.variable_scope("loss_function"):
        # 构造损失函数(均方误差)
        loss = tf.reduce_mean(tf.square(y_predict-y_true))

    with tf.variable_scope("optimizer"):
        # 优化损失(梯度下降法)
        optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)


    # 收集变量,在TensroBoard中显示
    tf.summary.scalar(name="loss",tensor=loss)
    tf.summary.histogram("weights",weights)
    tf.summary.histogram("bias",bias)
    # 合并变量
    merged = tf.summary.merge_all()

    # 创建Saver对象
    saver = tf.train.Saver()

    # 显式初始化变量
    init = tf.global_variables_initializer()

    # 开启会话
    with tf.Session() as sess:
        # 先初始化变量
        sess.run(init)

        # 创建事件文件
        file_writer = tf.summary.FileWriter("../tf_out/linear",graph=sess.graph)

        # 训练前的模型参数
        print("训练前的模型参数,权重：%f,偏置：%f,损失：%f" %(weights.eval(),bias.eval(),loss.eval()))

        # 开始训练
        for i in range(1000):
            sess.run(optimizer)
            print("第%d次训练后的模型参数,权重：%f,偏置：%f,损失：%f" % (i+1,weights.eval(), bias.eval(), loss.eval()))

            # 每次迭代需要收集变量
            summary = sess.run(merged)
            # 每次迭代后的变量写入事件文件
            file_writer.add_summary(summary,i)

            # 保存训练模型
            if i % 10 == 0:
                saver.save(sess,"../model_checkpoint/my_linear.ckpt")

    return None

def restore_model():
    with tf.variable_scope("prepare_data"):
        # 准备真实数据：100个样本
        x = tf.random.normal(shape=[100, 1], name="features")
        y_true = tf.matmul(x, [[0.8]]) + 0.7

    with tf.variable_scope("create_model"):
        # 构建模型
        # 　用变量定义模型参数
        weights = tf.Variable(initial_value=tf.random.normal(shape=[1, 1]), name="Weights")
        bias = tf.Variable(initial_value=tf.random.normal(shape=[1, 1]), name="Bias")
        y_predict = tf.matmul(x, weights) + bias

    with tf.variable_scope("loss_function"):
        # 构造损失函数(均方误差)
        loss = tf.reduce_mean(tf.square(y_predict - y_true))

    # 创建Saver对象
    saver = tf.train.Saver()

    with tf.Session() as sess:
        if os.path.exists("../model_checkpoint/checkpoint"):
            saver.restore(sess,"../model_checkpoint/my_linear.ckpt")
            print("训练后的模型参数,权重：%f,偏置：%f,损失：%f" % (weights.eval(), bias.eval(), loss.eval()))


if __name__ == "__main__":
    # linear_regression()
    # linear_regression2()
    restore_model()
