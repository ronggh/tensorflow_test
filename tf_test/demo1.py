import tensorflow as tf
# 这两行代码设置日志输出级别，可以屏蔽TF没有采用源码安装无法加速的日志输出
import os
os.environ['TF_CPP_MIN_LOOG-LEVEL']='2'

tf.app.flags

def tf_demo1():
    """
    用TF实现一个简单的加法运算
    :return: None
    """
    # 原生python实现的加法
    a = 2
    b = 3
    c = a + b
    print("原生python实现的加法结果：\n",c)

    # TF 实现的加法
    a_t = tf.constant(2)
    b_t = tf.constant(3)
    c_t = a_t + b_t
    print("TF实现一个简单的加法:\n",c_t)

    # 开启TF会话
    with tf.Session() as sess:
        c_t_value = sess.run(c_t)
        print("TF实现一个简单的加法结果:\n", c_t_value)

    return None

def graph_demo():
    """
    图的演示
    :return:
    """
    # TF 实现的加法
    a_t = tf.constant(2)
    b_t = tf.constant(3)
    c_t = a_t + b_t
    print("TF实现一个简单的加法:\n", c_t)

    # 查看默认图的方法,有两种：
    # 方法一：通过tf.get_default_graph()
    default_g = tf.get_default_graph()
    print("通过调用方法获取的默认图：\n",default_g)

    # 方法二：通过属性
    print("a_t的图属性：\n",a_t.graph)
    print("c_t的图属性：\n", c_t.graph)

    # 开启TF会话，无参默认在默认的图上开启
    with tf.Session() as sess:
        c_t_value = sess.run(c_t)
        print("TF实现一个简单的加法结果:\n", c_t_value)
        print("Session的图属性：\n",sess.graph)

    # 创建自定会义图
    new_g = tf.Graph()
    with new_g.as_default():
        # 可以指定指令名称
        a_new = tf.constant(10,name="a")
        b_new = tf.constant(20,name="b")
        c_new = tf.add(a_new,b_new,name="answer")

        print("c_new的图属性：\n",c_new.graph)
    # 开启new_g的会话
    with tf.Session(graph=new_g) as new_sess:
        c_new_value = new_sess.run(c_new)
        print("c_new_valule:\n",c_new_value)
        print("new_sess中自定义的图：\n",new_sess.graph)

        # 将自定义图序列化到本地，生成event文件
        tf.summary.FileWriter("../tf_out/summary",graph=new_sess.graph)
    return None

def session_demo():
    """
    会话演示
    :return:
    """
    t_a = tf.constant(10)
    t_b = tf.constant(20)
    t_c = tf.add(t_a,t_b)

    # 开启会话
    # 开启会话方式一：
    sess1 = tf.Session()
    c_value = sess1.run(t_c)
    print("c_value:\n",c_value)
    # 释放会话资源
    sess1.close()

    # 开启会话方式二：上下语，更常用,不需要显式释放,
    # 指定config参数，会打印出设备相关信息
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
        c_value = sess.run(t_c)
        print("c_value:\n", c_value)

    return None

def session_run_demo():
    """
    会话的run()方法
    :return:
    """
    # 定义占位符
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    sum_ab = tf.add(a, b)

    # 开启会话
    with tf.Session() as sess:
        print("占位符操作结果:\n", sess.run(sum_ab, feed_dict={a: 3.0, b: 2.0}))
    return None

def tensor_demo():
    """
    张量演示
    :return:
    """
    tensor1 = tf.constant(5.0)
    tensor2 = tf.constant([1,2,3,4])
    linear_squares = tf.constant([[1,2,3],[4,5,6],[7,8,9]],dtype=tf.int32)

    print("tensor1:\n",tensor1)
    print("tensor2:\n",tensor2)
    print("linear_squares:\n",linear_squares)


    # 定义固定和随机的张量
    zeros = tf.zeros(shape=[3,4])
    ones = tf.ones(shape=[3,4])
    ohters = tf.constant(3,shape=[3,4])
    randoms = tf.random.normal(shape=[3,4])

    # 张量的类型修改
    fls = tf.cast(linear_squares,tf.float32)

    # 静态张量改变形状
    # a_p,b_p是形状没有完全固定下来的张量，可以改变
    a_p = tf.placeholder(dtype=tf.float32,shape=[None,None])
    b_p = tf.placeholder(dtype=tf.float32,shape=[None,10])
    c_p = tf.placeholder(dtype=tf.float32,shape=[3,5])

    print("a_p:\n",a_p)
    print("b_p:\n",b_p)
    print("c_p:\n",c_p)

    # 静态修改形状，只能改未确定的部分
    #a_p.set_shape([3, 6])
    b_p.set_shape([5, 10])
    print("a_p:\n", a_p)
    print("b_p:\n", b_p)

    # 动态修改形状，会产生新的张量
    a_p_reshape = tf.reshape(a_p,shape=[3,4,2])
    print("a_p_reshape:\n",a_p_reshape)

    with tf.Session() as sess:
        print("zeros:\n",sess.run(zeros))
        print("ones:\n",sess.run(ones))
        print("ohters:\n",sess.run(ohters))
        print("ohters:\n",sess.run(randoms))
        print("类型转换后的张量:\n",sess.run(fls))

    return None

if __name__ =="__main__":
    # 用TF实现一个简单的加法运算
    # tf_demo1()
    # 图属性
    # graph_demo()
    # 会话演示
    # session_demo()
    # 会话的run()方法
    # session_run_demo()
    # 张量演示
    tensor_demo()