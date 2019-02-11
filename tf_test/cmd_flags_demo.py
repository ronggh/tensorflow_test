import tensorflow as tf
# 这两行代码设置日志输出级别，可以屏蔽TF没有采用源码安装无法加速的日志输出
import os
os.environ['TF_CPP_MIN_LOOG-LEVEL']='2'

# 定义一些命令行参数
tf.app.flags.DEFINE_integer("max_step",0,"训练模型的步数")
tf.app.flags.DEFINE_string("model_dir"," ","模型保存的路径+模型名字")

# 定义获取命令行参数
FLAGS = tf.app.flags.FLAGS

def cmd_flags_demo():
    """
    命令行参数演示
    :return:
    """
    print("max_step:\n",FLAGS.max_step)
    print("model_dir:\n",FLAGS.model_dir)

    return None

# main(argv)函数
def main(argv):
    print("max_step:\n", FLAGS.max_step)
    print("model_dir:\n", FLAGS.model_dir)

if __name__ == "__main__":
    # cmd_flags_demo()
    # tf.app.run()会自动调用main()函数
    tf.app.run()
