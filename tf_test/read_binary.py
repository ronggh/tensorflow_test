import tensorflow as tf
# 这两行代码设置日志输出级别，可以屏蔽TF没有采用源码安装无法加速的日志输出
import os
os.environ['TF_CPP_MIN_LOOG-LEVEL']='2'

# 读取二进制文件，CIFAR10
# https://www.cs.toronto.edu/~kriz/cifar.html
# 面向对象方式实现
class Cifar(object):
    def __init__(self):
        # 初始化操作
        # 定义图片
        self.height = 32
        self.width = 32
        self.channels = 3

        # 字节数
        # 图像
        self.image_bytes = self.height * self.width * self.channels
        # 标签
        self.label_bytes = 1
        self.all_bytes = self.image_bytes + self.label_bytes


    def read_and_decode(self,file_list):
        """
        读取二进制文件并解码
        :return:
        """
        # 构造文件名队列
        file_queue = tf.train.string_input_producer(file_list)
        # 读取与解码
        reader = tf.FixedLengthRecordReader(self.all_bytes)
        # key是文件名，value是一个样本数据
        key, value = reader.read(file_queue)
        decoded = tf.decode_raw(value, tf.uint8)
        print("key:\n",key)
        print("value:\n",value)
        print("decoded:\n",decoded)

        # 分割出目标值和特征值：label和image
        # label: 1个字节
        # image: 1024个R，1024个G，1024个B，共3072个RGB
        # 每个样本的shape = (channel, height, weight) = (3, 32, 32)
        # 需要转换成TF的表示习惯(height, weight, channel) = (32, 32, 3)
        # 进行切片处理
        label = tf.slice(decoded,[0],[self.label_bytes])
        image = tf.slice(decoded,[self.label_bytes],[self.image_bytes])
        print("label:\n",label)
        print("image:\n",image)

        # 调整图片形状
        image_reshape = tf.reshape(image,shape=[self.channels,self.height,self.width])
        print("image_reshape:\n", image_reshape)

        # 转换成TF格式，[height,weith,channels]
        image_trans = tf.transpose(image_reshape,[1,2,0])
        print("image_trans:\n", image_trans)
        # 调整为float32,提高计算精度,加入这行时，读取时也需要进行转换，否则会出错的
        #image_cast = tf.cast(image_trans,tf.float32)
        # print("image_cast:\n", image_cast)

        # 批处理
        label_batch,image_batch = tf.train.batch([label,image_trans],batch_size=100,num_threads=1,capacity=100)
        print("label_batch:\n", label_batch)
        print("image_batch:\n", image_batch)


        # 开启会话
        with tf.Session() as sess:
            # 开启线程
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)

            new_key,new_value,new_decoded,new_label,new_image,new_image_reshape = sess.run([key,value,decoded,label,image,image_reshape])
            label_value,image_value = sess.run([label_batch,image_batch])
            print("new_key:\n",new_key)
            print("new_value:\n",new_value)
            print("new_decoded:\n",new_decoded)
            print("new_label:\n",new_label)
            print("new_image:\n",new_image)
            print("new_image_reshape:\n",new_image_reshape)
            print("label_value:\n",label_value)
            print("image_value:\n",image_value)

            #  回收线程
            coord.request_stop()
            coord.join(threads)

        return image_value,label_value

    def write_to_tfrecords(self,image_batch,label_batch):
        """
        数据写入，TFRecords文件
        :return:
        """
        with tf.python_io.TFRecordWriter("../tf_out/cifar10.tfrecords") as writer:
            # 循环构造100个example实例，度序列化写入
            for i in range(100):
                image = image_batch[i].tostring()
                label = label_batch[i][0] # 直接取出值
                example = tf.train.Example(features=tf.train.Features(feature={
                    "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                }))
                # 将序列化后的example写入文件
                writer.write(example.SerializeToString())
        return None

    def read_tfrecords(self):
        """
        读取TFRecords文件
        :return:
        """
        #1、构造文件名队列
        file_queue = tf.train.string_input_producer(["../tf_out/cifar10.tfrecords"])

        #2、读取数据并进行解码
        #读取 -->  解析example --> 解码
        reader = tf.TFRecordReader()
        key,value = reader.read(file_queue)
        feature = tf.parse_single_example(value, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64)
        })
        image = feature["image"]
        label = feature["label"]
        # print("image:\n",image)
        # print("label:\n",label)

        # 需要解码image
        image_decoded = tf.decode_raw(image,tf.uint8)
        # 形状调整
        image_reshaped = tf.reshape(image_decoded,[self.height,self.width,self.channels])
        # print("image_decoded:\n",image_decoded)
        print("image_reshaped:\n",image_reshaped)

        #3、放入批处理队列
        label_batch,image_batch = tf.train.batch([label, image_reshaped], batch_size=100, num_threads=1, capacity=100)

        print("label_batch:\n", label_batch)
        print("image_batch:\n", image_batch)

        #4、开启会话线程运行
        # 开启会话
        with tf.Session() as sess:
            # 开启线程
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            image_value,label_value = sess.run([image,label])
            image_batch_value, label_batch_value = sess.run([image_batch,label_batch])

            print("image_value:\n",image_value)
            print("label_value:\n",label_value)

            print("image_batch_value:\n",image_batch_value)
            print("label_batch_value:\n",label_batch_value)

            # 回收线程
            coord.request_stop()
            coord.join(threads)

        return None

if __name__ == "__main__":
    # 读取文件名
    filenames = os.listdir("../tf_out/cifar10")
    # print("filenames:\n",filenames)
    # 构造路径+文件名，只需要.bin文件
    file_list = [os.path.join("../tf_out/cifar10/",file) for file in filenames if file[-3:]=="bin"]
    # print("file_list:\n",file_list)

    # 实例化Cifar
    cifar = Cifar()
    # 读文件，解码，转换
    # images,lebels = cifar.read_and_decode(file_list)

    # 写入到TFRecords文件
    # cifar.write_to_tfrecords(images,lebels)

    # 读取TFRecords文件
    cifar.read_tfrecords()


