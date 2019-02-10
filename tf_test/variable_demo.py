import tensorflow as tf
# 这两行代码设置日志输出级别，可以屏蔽TF没有采用源码安装无法加速的日志输出
import os
os.environ['TF_CPP_MIN_LOOG-LEVEL']='2'

def variable_demo():
    """
    变量演示
    :return:
    """
    # 定义
    a = tf.Variable(initial_value=30)
    b = tf.Variable(initial_value=40)
    sum = tf.add(a,b)

    print("a:\n",a)
    print("b:\n",b)
    print("sum:\n",sum)

    # 必须显式初始化变量，才能运行
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # 完成变量初始化
        sess.run(init)
        print("sum:\n",sess.run(sum))

    return None

def variable_demo2():
    """
    修改命名空间
    :return:
    """
    # 命名空间1：my_scope1
    with tf.variable_scope("my_scope1"):
        # 定义
        a = tf.Variable(initial_value=30)
        b = tf.Variable(initial_value=40)

    # 命名空间2：my_scope2
    with tf.variable_scope("my_scope2"):
        sum = tf.add(a, b)

    print("a:\n", a)
    print("b:\n", b)
    print("sum:\n", sum)

    # 必须显式初始化变量，才能运行
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # 完成变量初始化
        sess.run(init)
        print("sum:\n", sess.run(sum))

    return None

if __name__ =="__main__":
    # variable_demo()
    variable_demo2()