from tensorflow.examples.tutorials.mnist import input_data
import  tensorflow as tf
# 这两行代码设置日志输出级别，可以屏蔽TF没有采用源码安装无法加速的日志输出
import os
os.environ['TF_CPP_MIN_LOOG-LEVEL']='2'

def mnist_fc():
    """
    全连接网络实现手写数字识别
    :return:
    """
    # 读取数据
    path = "../mnist_data"
    mnist = input_data.read_data_sets(path,one_hot=True)

    # 图片28*28 = 784
    x = tf.placeholder(dtype=tf.float32,shape=[None,784])
    y_true = tf.placeholder(dtype=tf.float32,shape=[None,10])

    # 构造模型
    weights = tf.Variable(initial_value=tf.random_normal(shape=[784,10]))
    bias = tf.Variable(initial_value=tf.random_normal(shape=[10]))
    y_predict = tf.matmul(x,weights) + bias

    # 构造损失函数
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_predict)
    loss = tf.reduce_mean(diff)

    # 优化失函数
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    # 准确率计算
    # 判断两个位置是否一致，求出平均值
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
    mnist_fc()