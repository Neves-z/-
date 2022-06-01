
import pickle
import warnings

import numpy as np
import tensorflow as tf


warnings.filterwarnings("ignore")


class Text_reg:
    def __init__(self, texts, REG_PENALTY=0, is_training=True):
        self.texts = texts
        self.parameters = []
        self.resnet_texts(is_training)
        self.resnet_part()
        self.output = tf.compat.v1.nn.sigmoid(self.reg_head, name="output")
        self.cost_reg = REG_PENALTY * tf.compat.v1.reduce_mean(tf.compat.v1.square(self.parameters[-2])) / 2

    def batch_normalization(self, x, depth, is_training=True, name='conv0'):
        # 缩放  tf.ones_initializer 生成初始化为1的张量的初始化器
        gamma = tf.compat.v1.get_variable("gamma_" + name, [depth], initializer=tf.ones_initializer)
        # 样本偏移  tf.zeros_initializer 生成张量初始化为0的初始化器
        beta = tf.compat.v1.get_variable("beta_" + name, [depth], initializer=tf.zeros_initializer)
        # 均值
        pop_mean = tf.compat.v1.get_variable("mean_" + name, [depth], initializer=tf.zeros_initializer, trainable=False)
        # 方差
        pop_variance = tf.compat.v1.get_variable("variance_" + name, [depth], initializer=tf.ones_initializer,
                                                 trainable=False)
        self.parameters += [gamma, beta, pop_mean, pop_variance]
        if is_training:
            # 均值 方差
            batch_mean, batch_variance = tf.compat.v1.nn.moments(x, [0, 1, 2], keep_dims=False)
            decay = 0.99
            train_mean = tf.compat.v1.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_variance = tf.compat.v1.assign(pop_variance, pop_variance * decay + batch_variance * (1 - decay))
            with tf.control_dependencies([train_mean, train_variance]):
                return tf.nn.batch_normalization(x, batch_mean, batch_variance, beta, gamma, 1e-3)
        else:
            # out = gama*(x−mean) / var + beta  var=sqrt(variance +ε)
            return tf.nn.batch_normalization(x, pop_mean, pop_variance, beta, gamma, 1e-3)

    def resnet_texts(self,is_training):

        with tf.compat.v1.name_scope("conv_texts_0") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 49, 1, 32], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.texts, kernel, [1, 4, 4, 1], padding="SAME")
            out = self.batch_normalization(conv, 32, is_training, name=scope)
            self.conv_texts_0 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

            # pool1
        self.texts_pool1 = tf.compat.v1.nn.max_pool(
            self.conv_texts_0,
            ksize=[1, 1, 9, 1],
            strides=[1, 4, 4, 1],
            padding="SAME",
            name="texts_pool1",
        )

        # block 1
        # conv1_1
        with tf.compat.v1.name_scope("conv_texts1_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 32, 32], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.texts_pool1, kernel, [1, 1, 1, 1], padding="SAME")
            out = self.batch_normalization(conv, 32, is_training, name=scope)
            self.conv_texts1_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

        # conv1_2
        with tf.compat.v1.name_scope("conv_texts1_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 32, 32], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_texts1_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_texts1_2 = self.batch_normalization(conv, 32, is_training, name=scope)
            self.parameters += [kernel]

        self.conv_texts_1 = tf.nn.relu(self.texts_pool1 + self.conv_texts1_2, name=scope)

        # conv2_1
        with tf.compat.v1.name_scope("conv_texts2_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 32, 32], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_texts_1, kernel, [1, 1, 1, 1], padding="SAME")
            out = self.batch_normalization(conv, 32, is_training, name=scope)
            self.conv_texts2_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

        # conv2_2
        with tf.compat.v1.name_scope("conv_texts2_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 32, 32], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_texts2_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_texts2_2 = self.batch_normalization(conv, 32, is_training, name=scope)
            self.parameters += [kernel]

        self.conv_texts_2 = tf.nn.relu(self.conv_texts_1 + self.conv_texts2_2, name=scope)

        # block 2
        # conv3_1
        with tf.compat.v1.name_scope("conv_texts3_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 32, 32], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_texts_2, kernel, [1, 4, 4, 1], padding="SAME")
            out = self.batch_normalization(conv, 32, is_training, name=scope)
            self.conv_texts3_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

        # conv3_2
        with tf.compat.v1.name_scope("conv_ados3_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 32, 64], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_texts3_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_texts3_2 = self.batch_normalization(conv, 64, is_training, name=scope)
            self.parameters += [kernel]

        # conv3_3
        with tf.compat.v1.name_scope("conv_texts3_3") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 1, 32, 64], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_texts_2, kernel, [1, 4, 4, 1], padding="SAME")
            self.conv_texts3_3 = self.batch_normalization(conv, 64, is_training, name=scope)
            self.parameters += [kernel]

        self.conv_texts_3 = tf.nn.relu(self.conv_texts3_3 + self.conv_texts3_2, name=scope)

        # conv4_1
        with tf.compat.v1.name_scope("conv_texts4_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 64, 64], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_texts_3, kernel, [1, 1, 1, 1], padding="SAME")
            out = self.batch_normalization(conv, 64, is_training, name=scope)
            self.conv_texts4_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

        # conv4_2
        with tf.compat.v1.name_scope("conv_texts4_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 64, 64], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_texts4_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_texts4_2 = self.batch_normalization(conv, 64, is_training, name=scope)
            self.parameters += [kernel]

        self.conv_texts_4 = tf.compat.v1.nn.relu(self.conv_texts_3 + self.conv_texts4_2, name=scope)

        # block 3
        # conv5_1
        with tf.compat.v1.name_scope("conv_texts5_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 64, 64], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_texts_4, kernel, [1, 4, 4, 1], padding="SAME")
            out = self.batch_normalization(conv, 64, is_training, name=scope)
            self.conv_texts5_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

        # conv5_2
        with tf.compat.v1.name_scope("conv_texts5_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 64, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_texts5_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_texts5_2 = self.batch_normalization(conv, 128, is_training, name=scope)
            self.parameters += [kernel]

        # conv5_3 残差模块
        with tf.compat.v1.name_scope("conv_texts5_3") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 1, 64, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_texts_4, kernel, [1, 4, 4, 1], padding="SAME")
            self.conv_texts5_3 = self.batch_normalization(conv, 128, is_training, name=scope)
            self.parameters += [kernel]

        self.conv_texts_5 = tf.compat.v1.nn.relu(self.conv_texts5_2 + self.conv_texts5_3, name=scope)

        # conv6_1
        with tf.compat.v1.name_scope("conv_texts6_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 128, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_texts_5, kernel, [1, 1, 1, 1], padding="SAME")
            out = self.batch_normalization(conv, 128, is_training, name=scope)
            self.conv_texts6_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

        # conv6_2
        with tf.compat.v1.name_scope("conv_texts6_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 128, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_texts6_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_texts6_2 = self.batch_normalization(conv, 128, is_training, name=scope)
            self.parameters += [kernel]

        self.conv_texts_6 = tf.compat.v1.nn.relu(self.conv_texts_5 + self.conv_texts6_2, name=scope)

        # block 4
        # conv7_1
        with tf.compat.v1.name_scope("conv_texts7_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 128, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_texts_6, kernel, [1, 4, 4, 1], padding="SAME")
            out = self.batch_normalization(conv, 128, is_training, name=scope)
            self.conv_texts7_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

        # conv7_2
        with tf.compat.v1.name_scope("conv_texts7_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 128, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_texts7_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_texts7_2 = self.batch_normalization(conv, 256, is_training, name=scope)
            self.parameters += [kernel]

        # conv7_3 残差模块
        with tf.compat.v1.name_scope("conv_texts7_3") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 1, 128, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_texts_6, kernel, [1, 4, 4, 1], padding="SAME")
            self.conv_ados7_3 = self.batch_normalization(conv, 256, is_training, name=scope)
            self.parameters += [kernel]

        self.conv_texts_7 = tf.compat.v1.nn.relu(self.conv_texts7_2 + self.conv_ados7_3, name=scope)

        # conv8_1
        with tf.compat.v1.name_scope("conv_texts8_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 256, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_texts_7, kernel, [1, 1, 1, 1], padding="SAME")
            out = self.batch_normalization(conv, 256, is_training, name=scope)
            self.conv_texts8_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

        # conv8_2
        with tf.compat.v1.name_scope("conv_texts8_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 256, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_texts8_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_texts8_2 = self.batch_normalization(conv, 256, is_training, name=scope)
            self.parameters += [kernel]

        self.conv_texts_8 = tf.compat.v1.nn.relu(self.conv_texts_7 + self.conv_texts8_2, name=scope)

        height = int(self.conv_texts_8.get_shape().as_list()[1])  # =1
        weight = int(self.conv_texts_8.get_shape().as_list()[2])  # =11264/4/4/4/4/4=11
        # pool2
        # 全局平均池化
        self.texts_pool2 = tf.compat.v1.nn.avg_pool(
            self.conv_texts_8,
            ksize=[1, height, weight, 1],
            strides=[1, height, weight, 1],
            padding="VALID",
            name="texts_pool2",
        )



    def resnet_part(self):
        self.parameters = []
        # fc1
        with tf.compat.v1.name_scope("reg_head") as scope:

            #tensorflow查看形状XXX.get_shape()
            # np.prod 计算所谓元素的乘积  np.prod([1,2,3]=6
            shape = int(np.prod(self.texts_pool2.get_shape()[1:]))
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
            self.texts_flat = tf.compat.v1.nn.l2_normalize(
                tf.compat.v1.reshape(self.texts_pool2, [-1, int(shape)]), 1
            )

            # 添加偏置项 tk.nn.bias_add()
            # 函数：tf.matmul 表示：将矩阵 a 乘以矩阵 b,生成a * b
            self.reg_head = tf.compat.v1.nn.bias_add(
                tf.compat.v1.matmul(self.texts_flat, fc1w), fc1b, name=scope
            )
            self.parameters += [fc1w, fc1b]



    def load_trained_model(self, pickle_file, sess):
        with open(pickle_file, "rb") as pfile:
            param = pickle.load(pfile)
        for i in range(len(param)):
            sess.run(self.parameters[i].assign(param[i]))
