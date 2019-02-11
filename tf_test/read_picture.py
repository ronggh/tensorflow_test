import tensorflow as tf
# 这两行代码设置日志输出级别，可以屏蔽TF没有采用源码安装无法加速的日志输出
import os
os.environ['TF_CPP_MIN_LOOG-LEVEL']='2'

def read_pic():
    """
    读取图片
    :return:
    """
    #1、构造图片文件名队列
    filenames = os.listdir("../tf_out/dog")
    #print(filenames)
    # 拼接成路径+文件名格式
    file_list = [os.path.join("../tf_out/dog/",file) for file in filenames]
    #print(file_list)
    file_queue = tf.train.string_input_producer(file_list)

    #2、读取图片数据并进行解码
    reader = tf.WholeFileReader()
    # key是文件名，value是一张图片的原始编码形式
    key,value = reader.read(file_queue)
    # 解码
    image = tf.image.decode_jpeg(value)

    # 图像类型、大小处理
    image_resized = tf.image.resize_images(image,[200,200])
    # 彩色图片，3通道
    image_resized.set_shape([200,200,3])

    #3、处理图片数据形状，放入批处理队列
    image_batch = tf.train.batch([image_resized],batch_size=10,num_threads=1,capacity=10)

    #4、开启会话线程运行
    with tf.Session() as sess:
        # 开启线程
        # 先创建一个线程协调器
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        key_new, value_new, image_new, image_resized_new, image_batch_new = sess.run([key,value,image,image_resized,image_batch])
        print("key_new:\n",key_new)
        print("value_new:\n",value_new)
        print("image_new:\n",image_new)
        print("image_resized_new:\n",image_resized_new)
        print("image_batch_new:\n",image_batch_new)

        # 回收线程
        coord.request_stop()
        coord.join(threads)
    return None

if __name__ == "__main__":
    read_pic()