import tensorflow as tf
import glob
import captcha_setting
import os
import numpy as np


def read_pic():
    """
    读取图片文件
    :return:
    """
    # 1. 构造文件名队列
    # 使用glob获取文件名列表（也可以用os)
    filenames = glob.glob(captcha_setting.TRAIN_DATASET_PATH + "/*.png")
    # print("filenames:\n",filenames)
    file_queue = tf.train.string_input_producer(filenames)
    # 2. 读取与解码
    reader = tf.WholeFileReader()
    filename, image = reader.read(file_queue)
    # filename是全路径名，文件名前4个字母为图片中的校验码，image需要解码成三阶张量
    decoded_image = tf.image.decode_png(image)
    # 确定形状，方便批处理
    decoded_image.set_shape([captcha_setting.IMAGE_HEIGHT, captcha_setting.IMAGE_WIDTH, 3])
    # 修改精度
    decoded_image = tf.cast(decoded_image, tf.float32)
    # print("decoded_image:\n",decoded_image)

    # 3. 加入批处理
    filename_batch, image_batch = tf.train.batch([filename, decoded_image], batch_size=100, num_threads=2, capacity=100)

    return filename_batch, image_batch


def parse_filenames_to_labels(filenames):
    """
    解析文件名 ---> 校验码（标签） ---> []一行4列的张量
    NZPP_xxxxx.png ---> NZPP ---> [13,25,15,15]
    :return: 处理后的列表
    """

    labels = []
    for filename in filenames:
        # print(filename)
        # 取出4位的标签
        chars = str(filename).split(os.path.sep)[-1].split("_")[0]
        # print(chars)
        # 转换成[]
        label = []
        for c in chars:
            char_idx = captcha_setting.ALL_CHAR_SET.index(c)
            label.append(char_idx)
            # print(label)
        # print("\n")
        labels.append(label)

    # print(labels)
    return np.array(labels)


def create_weights(shape):
    """
    生成权重初始化值
    :param shape:
    :return:
    """
    return tf.Variable(initial_value=tf.random_normal(shape=shape, mean=0.0, stddev=0.1))


def create_cnn_model(x):
    """
    构造CNN网络模型
    两层卷积：卷积层、激活层、池化层
    全连接层：预测分类
    :param x: shape=[None,captcha_setting.IMAGE_HEIGHT,captcha_setting.IMAGE_WIDTH,3]
    :return:
    """
    # 1、第一个卷积大层
    with tf.variable_scope("conv-1"):
        # 1.1 卷积层：32个Filter，大小5*5，strides=1,padding="SAME"
        # 将x [None,784]进行形状转换成卷积函数要求的格式

        filter_conv1 = create_weights(shape=[5, 5, 3, 32])
        bias_conv1 = create_weights([32])
        features_conv1 = tf.nn.conv2d(input=x, filter=filter_conv1, strides=[1, 1, 1, 1], padding="SAME") + bias_conv1
        # 1.2 激活函数
        relu_conv1 = tf.nn.relu(features_conv1)
        # 1.3 池化层:大小2*2，strides=2
        pool_conv1 = tf.nn.max_pool(value=relu_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 2、第二个卷积大层
    with tf.variable_scope("conv-2"):
        # 2.1 卷积层：64个Filter，大小5*5，strides=1,padding="SAME"
        # [None,captcha_setting.IMAGE_HEIGHT,captcha_setting.IMAGE_WIDTH,3]
        # -->[None,captcha_setting.IMAGE_HEIGHT/2,captcha_setting.IMAGE_WIDTH/2,32]
        filter_conv2 = create_weights(shape=[5, 5, 32, 64])
        bias_conv2 = create_weights([64])
        features_conv2 = tf.nn.conv2d(input=pool_conv1, filter=filter_conv2, strides=[1, 1, 1, 1],
                                      padding="SAME") + bias_conv2

        # 2.2 激活函数
        relu_conv2 = tf.nn.relu(features_conv2)
        # 2.3 池化层:大小2*2，strides=2
        # [None,captcha_setting.IMAGE_HEIGHT/4,captcha_setting.IMAGE_WIDTH/4,32]
        pool_conv2 = tf.nn.max_pool(value=relu_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # 3、全连接层
    # tf.reshape()变成矩阵,[None, IMAGE_HEIGHT/4,IMAGE_WIDTH/4,32]
    # ---> [None, IMAGE_HEIGHT/4*IMAGE_WIDTH/4,64]，
    # 输出[None, 4*36]
    # [None, IMAGE_HEIGHT/4 * IMAGE_WIDTH/4 * 64] * [] = [None, 4*36]
    # 因此，weights = [IMAGE_HEIGHT/4 * IMAGE_WIDTH/4 * 64, 4*36], bias = [4*36]
    with tf.variable_scope("Full_Connection"):
        height = tf.cast(captcha_setting.IMAGE_HEIGHT / 4,tf.int32)
        width = tf.cast(captcha_setting.IMAGE_WIDTH / 4,tf.int32)
        x_fc = tf.reshape(pool_conv2, shape=[-1, height * width * 64])
        weights_fc = create_weights(shape=[height * width * 64, 4 * 36])
        bias_fc = create_weights(shape=[4 * 36])
        y_predict = tf.matmul(x_fc, weights_fc) + bias_fc

    return weights_fc,bias_fc,y_predict

def predict_result(x,weights,bias):
    """
    根据模型进行预测
    :param x:
    :param weights:
    :param bias:
    :return:
    """
    return tf.matmul(x, weights) + bias


if __name__ == "__main__":
    # 读取图片文件
    filename_batch, image_batch = read_pic()

    # 准备数据,彩色图片，3个通道
    x = tf.placeholder(dtype=tf.float32, shape=[None, captcha_setting.IMAGE_HEIGHT, captcha_setting.IMAGE_WIDTH, 3])
    # 校校码，4个长度，0-9以及A-Z
    y_true = tf.placeholder(dtype=tf.float32, shape=[None, 4 * captcha_setting.ALL_CHAR_SET_LEN])
    # 构造模型
    weights,bias,y_predict = create_cnn_model(x)
    # 构造损失函数
    loss_list = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_predict)
    loss = tf.reduce_mean(loss_list)
    # 优化损失函数
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    # 计算准确率
    equal_list = tf.reduce_all(
        tf.equal(tf.argmax(tf.reshape(y_true, shape=[-1, 4, 36]), axis=2),
                 tf.argmax(tf.reshape(y_predict, shape=[-1, 4, 36]), axis=2)), axis=1)
    accuracy = tf.reduce_mean(tf.cast(equal_list,tf.float32))

    # 初始化变量
    init = tf.global_variables_initializer()

    # 创建Saver对象
    saver = tf.train.Saver()

    # 是否为训练模式
    # is_training = True
    is_training = False

    # 开启会话
    with tf.Session() as sess:

        # 因为用到了队列，需要开启线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        filename_value, image_value = sess.run([filename_batch, image_batch])
        # print("filename_value:\n",filename_value)
        # print("image_value:\n",image_value)
        labels = parse_filenames_to_labels(filename_value)
        # print(labels)
        # labels转成one-hot编码
        labels_value = tf.reshape(tf.one_hot(labels, depth=captcha_setting.ALL_CHAR_SET_LEN), [-1, 4 * captcha_setting.ALL_CHAR_SET_LEN]).eval()

        # 开始训练
        if is_training:
            sess.run(init)
            for i in range(1000):
                _,loss_value,accuracy_value = sess.run([optimizer,loss,accuracy],feed_dict={x:image_value,y_true:labels_value})
                print("第%d次训练，损失为：%.2f,准确率：%.2f%%" % ((i+1),loss_value,accuracy_value*100))

                # 保存模型，退出
                if accuracy_value >= 1.0:
                    saver.save(sess, "../tf_out/model_checkpoint/captcha/captcha.ckpt")
                    print("weights:\n",weights.eval())
                    print("bias:\n",bias.eval())
                    break

        # 从保存的目录中恢复模型
        else:
            saver.restore(sess=sess,save_path="../tf_out/model_checkpoint/captcha/captcha.ckpt")
            weights_value,bias_value = sess.run([weights,bias])
            print("Weights:\n",weights_value)
            print("Bias:\n",bias_value)

            # 识别
            loss_value, accuracy_value = sess.run([loss, accuracy],
                                                     feed_dict={x: image_value, y_true: labels_value})

            print("识别，损失为：%.2f,准确率：%.2f%%" % ( loss_value, accuracy_value * 100))

        # 回收线程
        coord.request_stop()
        coord.join(threads)
