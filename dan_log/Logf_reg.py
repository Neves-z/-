
import pickle
import warnings

import numpy as np
import tensorflow as tf


warnings.filterwarnings("ignore")


class Audio_reg:
    def __init__(self, ados, REG_PENALTY=0):
        self.ados = ados
        self.fc()
        # sigmoid()
        self.output = tf.compat.v1.nn.sigmoid(self.reg_head, name="output")
        self.cost_reg = REG_PENALTY * tf.compat.v1.reduce_mean(tf.compat.v1.square(self.parameters[-2])) / 2



    def fc(self):
        self.parameters = []
        # fc1
        with tf.compat.v1.name_scope("reg_head") as scope:

            #tensorflow查看形状XXX.get_shape()
            # np.prod 计算所谓元素的乘积  np.prod([1,2,3)=6
            shape = int(np.prod(self.ados.get_shape()[1:]))
            # 定义权值变量   [shape, 5]最终输出为1*5的矩阵
            # tf.compat.v1.truncated_normal产生正态分布的随机数
            fc1w = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([shape, 5], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            fc1b = tf.compat.v1.Variable(
                tf.compat.v1.constant(1.0, shape=[5], dtype=tf.compat.v1.float32),
                trainable=True,
                name="biases",
            )
            self.audio_flat = tf.compat.v1.nn.l2_normalize(
                tf.compat.v1.reshape(self.ados, [-1, int(shape)]), 1
            )

            # 添加偏置项 tk.nn.bias_add()
            # 函数：tf.matmul 表示：将矩阵 a 乘以矩阵 b,生成a * b
            self.reg_head = tf.compat.v1.nn.bias_add(
                tf.compat.v1.matmul(self.audio_flat, fc1w), fc1b, name=scope
            )
            self.parameters += [fc1w, fc1b]



    def load_trained_model(self, pickle_file, sess):
        with open(pickle_file, "rb") as pfile:
            param = pickle.load(pfile)
        for i in range(len(param)):
            sess.run(self.parameters[i].assign(param[i]))
