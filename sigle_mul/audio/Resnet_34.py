import pickle
import warnings

import numpy as np
import tensorflow as tf
warnings.filterwarnings("ignore")


class audio_net:
    def __init__(self, ados, REG_PENALTY=0, is_training=True):
        self.ados = ados
        self.ado_parameters = []
        self.resnet_ados(is_training=is_training)
        self.net_part()
        self.output = tf.compat.v1.nn.sigmoid(self.reg_head, name="output")
        # self.ado_parameters[-2] = 获取列表倒数第二个元素 即全连接层weights
        self.cost_reg = REG_PENALTY * tf.compat.v1.reduce_mean(tf.compat.v1.square(self.ado_parameters[-2])) / 2

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
        self.ado_parameters += [gamma, beta, pop_mean, pop_variance]
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


    def resnet_ados(self, is_training):

        with tf.compat.v1.name_scope("conv_ados_0") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 49, 1, 64], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.ados, kernel, [1, 4, 4, 1], padding="SAME")
            out = self.batch_normalization(conv, 64, is_training, name=scope)
            self.conv_ados_0 = tf.compat.v1.nn.relu(out, name=scope)
            self.ado_parameters += [kernel]

        # pool1
        self.audio_pool1 = tf.compat.v1.nn.max_pool(
            self.conv_ados_0,
            ksize=[1, 1, 9, 1],
            strides=[1, 4, 4, 1],
            padding="SAME",
            name="audio_pool1",
        )

        # conv1_1
        with tf.compat.v1.name_scope("conv_ados1_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 64, 64], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.audio_pool1, kernel, [1, 1, 1, 1], padding="SAME")
            out = self.batch_normalization(conv, 64, is_training, name=scope)
            self.conv_ados1_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.ado_parameters += [kernel]

        # conv1_2
        with tf.compat.v1.name_scope("conv_ados1_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 64, 64], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados1_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados1_2 = self.batch_normalization(conv, 64, is_training, name=scope)
            self.ado_parameters += [kernel]

        self.conv_ados_1 = tf.nn.relu(self.audio_pool1 + self.conv_ados1_2, name=scope)

        # conv2_1
        with tf.compat.v1.name_scope("conv_ados2_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 64, 64], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_1, kernel, [1, 1, 1, 1], padding="SAME")
            out = self.batch_normalization(conv, 64, is_training, name=scope)
            self.conv_ados2_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.ado_parameters += [kernel]

        # conv2_2
        with tf.compat.v1.name_scope("conv_ados2_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 64, 64], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados2_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados2_2 = self.batch_normalization(conv, 64, is_training, name=scope)
            self.ado_parameters += [kernel]

        self.conv_ados_2 = tf.nn.relu(self.conv_ados_1 + self.conv_ados2_2, name=scope)

        # conv3_1
        with tf.compat.v1.name_scope("conv_ados3_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 64, 64], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_2, kernel, [1, 1, 1, 1], padding="SAME")
            out = self.batch_normalization(conv, 64, is_training, name=scope)
            self.conv_ados3_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.ado_parameters += [kernel]

        # conv3_2
        with tf.compat.v1.name_scope("conv_ados3_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 64, 64], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados3_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados3_2 = self.batch_normalization(conv, 64, is_training, name=scope)
            self.ado_parameters += [kernel]

        self.conv_ados_3 = tf.nn.relu(self.conv_ados_2 + self.conv_ados3_2, name=scope)

        # conv4_1
        with tf.compat.v1.name_scope("conv_ados4_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 64, 64], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_3, kernel, [1, 4, 4, 1], padding="SAME")
            out = self.batch_normalization(conv, 64, is_training, name=scope)
            self.conv_ados4_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.ado_parameters += [kernel]

        # conv4_2
        with tf.compat.v1.name_scope("conv_ados4_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 64, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados4_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados4_2 = self.batch_normalization(conv, 128, is_training, name=scope)
            self.ado_parameters += [kernel]

        # conv4_3
        with tf.compat.v1.name_scope("conv_ados4_3") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 64, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_3, kernel, [1, 4, 4, 1], padding="SAME")
            self.conv_ados4_3 = self.batch_normalization(conv, 128, is_training, name=scope)
            self.ado_parameters += [kernel]

        self.conv_ados_4 = tf.compat.v1.nn.relu(self.conv_ados4_3 + self.conv_ados4_2, name=scope)

        # conv5_1
        with tf.compat.v1.name_scope("conv_ados5_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 128, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_4, kernel, [1, 1, 1, 1], padding="SAME")
            out = self.batch_normalization(conv, 128, is_training, name=scope)
            self.conv_ados5_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.ado_parameters += [kernel]

        # conv5_2
        with tf.compat.v1.name_scope("conv_ados5_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 128, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados5_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados5_2 = self.batch_normalization(conv, 128, is_training, name=scope)
            self.ado_parameters += [kernel]

        self.conv_ados_5 = tf.compat.v1.nn.relu(self.conv_ados_4 + self.conv_ados5_2, name=scope)

        # conv6_1
        with tf.compat.v1.name_scope("conv_ados6_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 128, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_5, kernel, [1, 1, 1, 1], padding="SAME")
            out = self.batch_normalization(conv, 128, is_training, name=scope)
            self.conv_ados6_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.ado_parameters += [kernel]

        # conv6_2
        with tf.compat.v1.name_scope("conv_ados6_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 128, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados6_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados6_2 = self.batch_normalization(conv, 128, is_training, name=scope)
            self.ado_parameters += [kernel]

        self.conv_ados_6 = tf.compat.v1.nn.relu(self.conv_ados_5 + self.conv_ados6_2, name=scope)

        # conv7_1
        with tf.compat.v1.name_scope("conv_ados7_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 128, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_6, kernel, [1, 1, 1, 1], padding="SAME")
            out = self.batch_normalization(conv, 128, is_training, name=scope)
            self.conv_ados7_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.ado_parameters += [kernel]

        # conv7_2
        with tf.compat.v1.name_scope("conv_ados7_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 128, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados7_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados7_2 = self.batch_normalization(conv, 128, is_training, name=scope)
            self.ado_parameters += [kernel]

        self.conv_ados_7 = tf.compat.v1.nn.relu(self.conv_ados_6 + self.conv_ados7_2, name=scope)

        # conv8_1
        with tf.compat.v1.name_scope("conv_ados8_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 128, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_7, kernel, [1, 4, 4, 1], padding="SAME")
            out = self.batch_normalization(conv, 128, is_training, name=scope)
            self.conv_ados8_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.ado_parameters += [kernel]

        # conv8_2
        with tf.compat.v1.name_scope("conv_ados8_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 128, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados8_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados8_2 = self.batch_normalization(conv, 256, is_training, name=scope)
            self.ado_parameters += [kernel]

        # conv8_3
        with tf.compat.v1.name_scope("conv_ados8_3") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 128, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_7, kernel, [1, 4, 4, 1], padding="SAME")
            self.conv_ados8_3 = self.batch_normalization(conv, 256, is_training, name=scope)
            self.ado_parameters += [kernel]

        self.conv_ados_8 = tf.compat.v1.nn.relu(self.conv_ados8_2 + self.conv_ados8_3, name=scope)

        # conv9_1
        with tf.compat.v1.name_scope("conv_ados9_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 256, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_8, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados9_1 = self.batch_normalization(conv, 256, is_training, name=scope)
            self.ado_parameters += [kernel]

        # conv9_2
        with tf.compat.v1.name_scope("conv_ados9_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 256, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados9_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados9_2 = self.batch_normalization(conv, 256, is_training, name=scope)
            self.ado_parameters += [kernel]

        self.conv_ados_9 = tf.compat.v1.nn.relu(self.conv_ados_8 + self.conv_ados9_2, name=scope)

        # conv10_1
        with tf.compat.v1.name_scope("conv_ados10_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 256, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_9, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados10_1 = self.batch_normalization(conv, 256, is_training, name=scope)
            self.ado_parameters += [kernel]

        # conv10_2
        with tf.compat.v1.name_scope("conv_ados10_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 256, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados10_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados10_2 = self.batch_normalization(conv, 256, is_training, name=scope)
            self.ado_parameters += [kernel]

        self.conv_ados_10 = tf.compat.v1.nn.relu(self.conv_ados_9 + self.conv_ados10_2, name=scope)

        # conv11_1
        with tf.compat.v1.name_scope("conv_ados11_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 256, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_10, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados11_1 = self.batch_normalization(conv, 256, is_training, name=scope)
            self.ado_parameters += [kernel]

        # conv11_2
        with tf.compat.v1.name_scope("conv_ados11_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 256, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados11_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados11_2 = self.batch_normalization(conv, 256, is_training, name=scope)
            self.ado_parameters += [kernel]

        self.conv_ados_11 = tf.compat.v1.nn.relu(self.conv_ados_10 + self.conv_ados11_2, name=scope)

        # conv12_1
        with tf.compat.v1.name_scope("conv_ados12_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 256, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_11, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados12_1 = self.batch_normalization(conv, 256, is_training, name=scope)
            self.ado_parameters += [kernel]

        # conv12_2
        with tf.compat.v1.name_scope("conv_ados12_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 256, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados12_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados12_2 = self.batch_normalization(conv, 256, is_training, name=scope)
            self.ado_parameters += [kernel]

        self.conv_ados_12 = tf.compat.v1.nn.relu(self.conv_ados_11 + self.conv_ados12_2, name=scope)

        # conv13_1
        with tf.compat.v1.name_scope("conv_ados13_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 256, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_12, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados13_1 = self.batch_normalization(conv, 256, is_training, name=scope)
            self.ado_parameters += [kernel]

        # conv13_2
        with tf.compat.v1.name_scope("conv_ados13_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 256, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados13_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados13_2 = self.batch_normalization(conv, 256, is_training, name=scope)
            self.ado_parameters += [kernel]

        self.conv_ados_13 = tf.compat.v1.nn.relu(self.conv_ados_12 + self.conv_ados13_2, name=scope)

        # conv14_1
        with tf.compat.v1.name_scope("conv_ados14_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 256, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_13, kernel, [1, 4, 4, 1], padding="SAME")
            self.conv_ados14_1 = self.batch_normalization(conv, 256, is_training, name=scope)
            self.ado_parameters += [kernel]

        # conv14_2
        with tf.compat.v1.name_scope("conv_ados14_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 256, 512], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados14_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados14_2 = self.batch_normalization(conv, 512, is_training, name=scope)
            self.ado_parameters += [kernel]

        # conv14_3
        with tf.compat.v1.name_scope("conv_ados14_3") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 256, 512], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_13, kernel, [1, 4, 4, 1], padding="SAME")
            self.conv_ados14_3 = self.batch_normalization(conv, 512, is_training, name=scope)
            self.ado_parameters += [kernel]

        self.conv_ados_14 = tf.compat.v1.nn.relu(self.conv_ados14_2 + self.conv_ados14_3, name=scope)

        # conv15_1
        with tf.compat.v1.name_scope("conv_ados15_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 512, 512], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_14, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados15_1 = self.batch_normalization(conv, 512, is_training, name=scope)
            self.ado_parameters += [kernel]

        # conv15_2
        with tf.compat.v1.name_scope("conv_ados15_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 512, 512], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados15_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados15_2 = self.batch_normalization(conv, 512, is_training, name=scope)
            self.ado_parameters += [kernel]

        self.conv_ados_15 = tf.compat.v1.nn.relu(self.conv_ados_14 + self.conv_ados15_2, name=scope)

        # conv16_1
        with tf.compat.v1.name_scope("conv_ados16_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 512, 512], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_15, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados16_1 = self.batch_normalization(conv, 512, is_training, name=scope)
            self.ado_parameters += [kernel]

        # conv16_2
        with tf.compat.v1.name_scope("conv_ados15_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 512, 512], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados16_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados16_2 = self.batch_normalization(conv, 512, is_training, name=scope)
            self.ado_parameters += [kernel]

        self.conv_ados_16 = tf.compat.v1.nn.relu(self.conv_ados_15 + self.conv_ados16_2, name=scope)

        height = int(self.conv_ados_16.get_shape().as_list()[1])  # =1
        weight = int(self.conv_ados_16.get_shape().as_list()[2])  # 324
        # print(height, weight)
        # pool2
        # 全局平均池化
        self.audio_pool2 = tf.compat.v1.nn.avg_pool(
            self.conv_ados_16,
            ksize=[1, height, weight, 1],
            strides=[1, height, weight, 1],
            padding="VALID",
            name="audio_pool2",
        )

        self.audio_pool3 = tf.compat.v1.nn.max_pool(
            self.conv_ados_16,
            ksize=[1, height, weight, 1],
            strides=[1, height, weight, 1],
            padding="VALID",
            name="audio_pool3",
        )

    def net_part(self):
        # fc1
        with tf.compat.v1.name_scope("reg_head") as scope:

            shape = 2 * int(np.prod(self.audio_pool2.get_shape()[1:]))   # 512*2 = 1024

            fc1w = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([shape, 5], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            fc1b = tf.compat.v1.Variable(
                tf.compat.v1.constant(1.0, shape=[5], dtype=tf.compat.v1.float32),
                trainable=True,
                name="biases",
            )


            self.audio_pool2_flat = tf.compat.v1.nn.l2_normalize(
                tf.compat.v1.reshape(self.audio_pool2, [-1, int(shape / 2)]), 1
            )
            self.audio_pool3_flat = tf.compat.v1.nn.l2_normalize(
                tf.compat.v1.reshape(self.audio_pool3, [-1, int(shape / 2)]), 1
            )
            # 连接 1024-d
            self.concat = tf.compat.v1.concat([self.audio_pool2_flat, self.audio_pool3_flat], 1)

            self.reg_head = tf.compat.v1.nn.bias_add(
                tf.compat.v1.matmul(self.concat, fc1w), fc1b, name=scope
            )
            self.ado_parameters += [fc1w, fc1b]


    def load_trained_model(self, pickle_file, sess):
        with open(pickle_file, "rb") as pfile:
            param = pickle.load(pfile)
        for i in range(len(param)):
            sess.run(self.ado_parameters[i].assign(param[i]))


