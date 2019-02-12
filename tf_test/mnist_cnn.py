from tensorflow.examples.tutorials.mnist import input_data
import  tensorflow as tf
# 这两行代码设置日志输出级别，可以屏蔽TF没有采用源码安装无法加速的日志输出
import os
os.environ['TF_CPP_MIN_LOOG-LEVEL']='2'

def create_weights(shape):
    """
    生成权重初始化值
    :param shape:
    :return:
    """
    return tf.Variable(initial_value=tf.random_normal(shape=shape,mean=0.0,stddev=0.1))

def create_cnn_model(x):
    """
    构造CNN网络模型
    两层卷积：卷积层、激活层、池化层
    全连接层：预测分类
    :param x:
    :return:
    """
    # 1、第一个卷积大层
    with tf.variable_scope("conv-1"):
        # 1.1 卷积层：32个Filter，大小5*5，strides=1,padding="SAME"
        # 将x [None,784]进行形状转换成卷积函数要求的格式
        input_x = tf.reshape(x,shape=[-1,28,28,1])

        filter_conv1 = create_weights(shape=[5,5,1,32])
        bias_conv1 = create_weights([32])
        features_conv1 = tf.nn.conv2d(input=input_x,filter=filter_conv1,strides=[1,1,1,1],padding="SAME") + bias_conv1
        # 1.2 激活函数
        relu_conv1 = tf.nn.relu(features_conv1)
        # 1.3 池化层:大小2*2，strides=2
        pool_conv1 = tf.nn.max_pool(value=relu_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    # 2、第二个卷积大层
    with tf.variable_scope("conv-2"):
        # 2.1 卷积层：64个Filter，大小5*5，strides=1,padding="SAME"
        filter_conv2 = create_weights(shape=[5, 5, 32, 64])
        bias_conv2 = create_weights([64])
        features_conv2 = tf.nn.conv2d(input=pool_conv1, filter=filter_conv2, strides=[1, 1, 1, 1],padding="SAME") + bias_conv2

        # 2.2 激活函数
        relu_conv2 = tf.nn.relu(features_conv2)
        # 2.3 池化层:大小2*2，strides=2
        pool_conv2 = tf.nn.max_pool(value=relu_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # 3、全连接层
    # tf.reshape()变成矩阵,[None, 7, 7, 64] - --> [None, 7 * 7 * 64]，
    # 最终输出0 ~9 的10个分类, 即[None, 10]
    # [None, 7 * 7 * 64] * [] = [None, 10]
    # 因此，weights = [7 * 64 * 64, 10], bias = [10]
    with tf.variable_scope("Full_Connection"):
        x_fc = tf.reshape(pool_conv2,shape=[-1,7 * 7 * 64])
        weights_fc = create_weights(shape=[7*7*64,10])
        bias_fc = create_weights(shape=[10])
        y_predict = tf.matmul(x_fc, weights_fc) + bias_fc

    return y_predict

def mnist_cnn():
    """
    CNN实现手写数字识别
    :return:
    """
    # 读取数据
    with tf.variable_scope("mnist_data"):
        path = "../mnist_data"
        mnist = input_data.read_data_sets(path,one_hot=True)

        # 图片28*28 = 784
        x = tf.placeholder(dtype=tf.float32,shape=[None,784])
        y_true = tf.placeholder(dtype=tf.float32,shape=[None,10])

    # 构造模型
    with tf.variable_scope("cnn_model"):
        y_predict = create_cnn_model(x)

    # 构造损失函数
    with tf.variable_scope("softmax_cross_entropy"):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_predict)
        loss = tf.reduce_mean(diff)

    # 优化损失函数
    with tf.variable_scope("optimizer"):
        # learning_rate=0.001效果较好，0.1，0.01效果较差
        # 梯度下降优化器
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
        # 亚当优化器
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # 准确率计算
    # 判断两个位置是否一致，求出平均值
    with tf.variable_scope("accuracy"):
        equal_list = tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_predict, axis=1))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 初始化变量
    init = tf.global_variables_initializer()

    # 开启会话
    with tf.Session() as  sess:
        sess.run(init)
        # 填充数据
        image,label = mnist.train.next_batch(100)
        print("训练前损失：%f\n" % sess.run(loss,feed_dict={x:image,y_true:label}))

        # 开始训练
        for i in range(3000):
            _,loss_value,accuracy_value = sess.run([optimizer,loss,accuracy],feed_dict={x:image,y_true:label})
            print("第%d次训练的损失：%f，准确率：%f\n" % ((i+1),loss_value,accuracy_value))


    return None

if __name__ == "__main__":
    mnist_cnn()