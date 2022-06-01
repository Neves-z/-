
import pickle
import warnings

import numpy as np
import tensorflow as tf


warnings.filterwarnings("ignore")


class  Resnet_18:
    def __init__(self, imgs, ados, REG_PENALTY=0, is_training=True):
        self.imgs = imgs
        self.ados = ados
        self.parameters = []
        self.resnet_imgs(is_training)
        self.resnet_ados(is_training)
        self.resnet_part()
        self.output = tf.compat.v1.nn.sigmoid(self.reg_head, name="output")
        self.cost_reg = REG_PENALTY * tf.compat.v1.reduce_mean(tf.compat.v1.square(self.parameters[-2])) / 2

    def batch_normalization(self, x, depth, is_training=True, name='conv0'):
        # 缩放  tf.ones_initializer 生成初始化为1的张量的初始化器
        gamma = tf.compat.v1.get_variable("gamma_"+name, [depth], initializer=tf.ones_initializer)
        # 样本偏移  tf.zeros_initializer 生成张量初始化为0的初始化器
        beta = tf.compat.v1.get_variable("beta_"+name, [depth], initializer=tf.zeros_initializer)
        # 均值
        pop_mean = tf.compat.v1.get_variable("mean_"+name, [depth], initializer=tf.zeros_initializer, trainable=False)
        # 方差
        pop_variance = tf.compat.v1.get_variable("variance_"+name, [depth], initializer=tf.ones_initializer, trainable=False)
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


    def resnet_imgs(self,is_training):

        with tf.compat.v1.name_scope("conv0") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([7, 7, 3, 32], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.imgs, kernel, [1, 2, 2, 1], padding="SAME")
            out = self.batch_normalization(conv, 32, is_training, name=scope)
            self.conv_0 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]


        # pool1
        self.pool1 = tf.compat.v1.nn.max_pool(
            self.conv_0,
            ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding="SAME",
            name="pool1",
        )

        # block 1
        # conv1_1
        with tf.compat.v1.name_scope("conv1_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 32, 32], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding="SAME")
            out = self.batch_normalization(conv, 32, is_training, name=scope)
            self.conv1_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

        # conv1_2
        with tf.compat.v1.name_scope("conv1_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 32, 32], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv1_2= self.batch_normalization(conv, 32, is_training, name=scope)
            self.parameters += [kernel]

        self.conv_1 = tf.nn.relu(self.pool1 + self.conv1_2, name=scope)

        # conv2_1
        with tf.compat.v1.name_scope("conv2_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 32, 32], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_1, kernel, [1, 1, 1, 1], padding="SAME")
            out = self.batch_normalization(conv, 32, is_training, name=scope)
            self.conv2_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

        # conv2_2
        with tf.compat.v1.name_scope("conv2_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 32, 32], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv2_2 = self.batch_normalization(conv, 32, is_training, name=scope)
            self.parameters += [kernel]

        self.conv_2 = tf.nn.relu(self.conv_1 + self.conv2_2, name=scope)

        # block 2
        # conv3_1
        with tf.compat.v1.name_scope("conv3_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 32, 32], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_2, kernel, [1, 2, 2, 1], padding="SAME")
            out = self.batch_normalization(conv, 32, is_training, name=scope)
            self.conv3_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

        # conv3_2
        with tf.compat.v1.name_scope("conv3_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 32, 64], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv3_2 = self.batch_normalization(conv, 64, is_training, name=scope)
            self.parameters += [kernel]

        # conv3_3
        with tf.compat.v1.name_scope("conv3_3") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 1, 32, 64], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_2, kernel, [1, 2, 2, 1], padding="SAME")
            self.conv3_3 = self.batch_normalization(conv, 64, is_training, name=scope)
            self.parameters += [kernel]

        self.conv_3 = tf.nn.relu(self.conv3_3 + self.conv3_2, name=scope)

        # conv4_1
        with tf.compat.v1.name_scope("conv4_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 64, 64], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_3, kernel, [1, 1, 1, 1], padding="SAME")
            out = self.batch_normalization(conv, 64, is_training, name=scope)
            self.conv4_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

        # conv4_2
        with tf.compat.v1.name_scope("conv4_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 64, 64], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv4_2 = self.batch_normalization(conv, 64, is_training, name=scope)
            self.parameters += [kernel]

        self.conv_4 = tf.compat.v1.nn.relu(self.conv_3+self.conv4_2, name=scope)



        # block 3
        # conv5_1
        with tf.compat.v1.name_scope("conv5_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 64, 64], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_4, kernel, [1, 2, 2, 1], padding="SAME")
            out = self.batch_normalization(conv, 64, is_training, name=scope)
            self.conv5_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

        # conv5_2
        with tf.compat.v1.name_scope("conv5_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 64, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv5_2 = self.batch_normalization(conv, 128, is_training, name=scope)
            self.parameters += [kernel]

        # conv5_3 残差模块
        with tf.compat.v1.name_scope("conv5_3") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 1, 64, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_4, kernel, [1, 2, 2, 1], padding="SAME")
            self.conv5_3 = self.batch_normalization(conv, 128, is_training, name=scope)
            self.parameters += [kernel]

        self.conv_5 = tf.compat.v1.nn.relu(self.conv5_2+self.conv5_3, name=scope)

        # conv6_1
        with tf.compat.v1.name_scope("conv6_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 128, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_5, kernel, [1, 1, 1, 1], padding="SAME")
            out = self.batch_normalization(conv, 128, is_training, name=scope)
            self.conv6_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

        # conv6_2
        with tf.compat.v1.name_scope("conv6_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 128, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv6_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv6_2 = self.batch_normalization(conv, 128, is_training, name=scope)
            self.parameters += [kernel]


        self.conv_6 = tf.compat.v1.nn.relu(self.conv_5 + self.conv6_2, name=scope)

        # block 4
        # conv7_1
        with tf.compat.v1.name_scope("conv7_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 128, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_6, kernel, [1, 2, 2, 1], padding="SAME")
            out = self.batch_normalization(conv, 128, is_training, name=scope)
            self.conv7_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

        # conv7_2
        with tf.compat.v1.name_scope("conv7_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 128, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv7_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv7_2 = self.batch_normalization(conv, 256, is_training, name=scope)
            self.parameters += [kernel]

        # conv7_3 残差模块
        with tf.compat.v1.name_scope("conv7_3") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 1, 128, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_6, kernel, [1, 2, 2, 1], padding="SAME")
            self.conv7_3 = self.batch_normalization(conv, 256, is_training, name=scope)
            self.parameters += [kernel]

        self.conv_7 = tf.compat.v1.nn.relu(self.conv7_2 + self.conv7_3, name=scope)

        # conv8_1
        with tf.compat.v1.name_scope("conv8_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 256, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_7, kernel, [1, 1, 1, 1], padding="SAME")
            out = self.batch_normalization(conv, 256, is_training, name=scope)
            self.conv8_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

        # conv8_2
        with tf.compat.v1.name_scope("conv8_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 256, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv8_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv8_2 = self.batch_normalization(conv, 256, is_training, name=scope)
            self.parameters += [kernel]

        self.conv_8 = tf.compat.v1.nn.relu(self.conv_7 + self.conv8_2, name=scope)


        #height = int(self.conv_8.get_shape().as_list()[0])  #=7
        #weight = int(self.conv_8.get_shape().as_list()[1])   #=7
        print(self.conv_8.get_shape().as_list()[1])
        print(self.conv_8.get_shape().as_list()[2])
        height = 7
        weight = 7
        # pool2
        # 全局平均池化
        self.pool2 = tf.compat.v1.nn.avg_pool(
            self.conv_8,
            ksize=[1, height, weight, 1],
            strides=[1, height, weight, 1],
            padding="VALID",
            name="pool2",
        )
    def resnet_ados(self, is_training):

        with tf.compat.v1.name_scope("conv_ados_0") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 49, 1, 32], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.ados, kernel, [1, 4, 4, 1], padding="SAME")
            out = self.batch_normalization(conv, 32, is_training, name=scope)
            self.conv_ados_0 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

        # pool1
        self.audio_pool1 = tf.compat.v1.nn.max_pool(
            self.conv_ados_0,
            ksize=[1, 1, 9, 1],
            strides=[1, 4, 4, 1],
            padding="SAME",
            name="audio_pool1",
        )

        # block 1
        # conv1_1
        with tf.compat.v1.name_scope("conv_ados1_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 32, 32], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.audio_pool1, kernel, [1, 1, 1, 1], padding="SAME")
            out = self.batch_normalization(conv, 32, is_training, name=scope)
            self.conv_ados1_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

        # conv1_2
        with tf.compat.v1.name_scope("conv_ados1_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 32, 32], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados1_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados1_2 = self.batch_normalization(conv, 32, is_training, name=scope)
            self.parameters += [kernel]

        self.conv_ados_1 = tf.nn.relu(self.audio_pool1 + self.conv_ados1_2, name=scope)

        # conv2_1
        with tf.compat.v1.name_scope("conv_ados2_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 32, 32], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_1, kernel, [1, 1, 1, 1], padding="SAME")
            out = self.batch_normalization(conv, 32, is_training, name=scope)
            self.conv_ados2_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

        # conv2_2
        with tf.compat.v1.name_scope("conv_ados2_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 32, 32], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados2_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados2_2 = self.batch_normalization(conv, 32, is_training, name=scope)
            self.parameters += [kernel]

        self.conv_ados_2 = tf.nn.relu(self.conv_ados_1 + self.conv_ados2_2, name=scope)

        # block 2
        # conv3_1
        with tf.compat.v1.name_scope("conv_ados3_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 32, 32], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_2, kernel, [1, 4, 4, 1], padding="SAME")
            out = self.batch_normalization(conv, 32, is_training, name=scope)
            self.conv_ados3_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

        # conv3_2
        with tf.compat.v1.name_scope("conv_ados3_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 32, 64], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados3_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados3_2 = self.batch_normalization(conv, 64, is_training, name=scope)
            self.parameters += [kernel]

        # conv3_3
        with tf.compat.v1.name_scope("conv_ados3_3") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 1, 32, 64], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_2, kernel, [1, 4, 4, 1], padding="SAME")
            self.conv_ados3_3 = self.batch_normalization(conv, 64, is_training, name=scope)
            self.parameters += [kernel]

        self.conv_ados_3 = tf.nn.relu(self.conv_ados3_3 + self.conv_ados3_2, name=scope)

        # conv4_1
        with tf.compat.v1.name_scope("conv_ados4_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 64, 64], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_3, kernel, [1, 1, 1, 1], padding="SAME")
            out = self.batch_normalization(conv, 64, is_training, name=scope)
            self.conv_ados4_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

        # conv4_2
        with tf.compat.v1.name_scope("conv_ados4_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 64, 64], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados4_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados4_2 = self.batch_normalization(conv, 64, is_training, name=scope)
            self.parameters += [kernel]

        self.conv_ados_4 = tf.compat.v1.nn.relu(self.conv_ados_3 + self.conv_ados4_2, name=scope)

        # block 3
        # conv5_1
        with tf.compat.v1.name_scope("conv_ados5_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 64, 64], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_4, kernel, [1, 4, 4, 1], padding="SAME")
            out = self.batch_normalization(conv, 64, is_training, name=scope)
            self.conv_ados5_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

        # conv5_2
        with tf.compat.v1.name_scope("conv_ados5_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 64, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados5_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados5_2 = self.batch_normalization(conv, 128, is_training, name=scope)
            self.parameters += [kernel]

        # conv5_3 残差模块
        with tf.compat.v1.name_scope("conv_ados5_3") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 1, 64, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_4, kernel, [1, 4, 4, 1], padding="SAME")
            self.conv_ados5_3 = self.batch_normalization(conv, 128, is_training, name=scope)
            self.parameters += [kernel]

        self.conv_ados_5 = tf.compat.v1.nn.relu(self.conv_ados5_2 + self.conv_ados5_3, name=scope)

        # conv6_1
        with tf.compat.v1.name_scope("conv_ados6_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 128, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_5, kernel, [1, 1, 1, 1], padding="SAME")
            out = self.batch_normalization(conv, 128, is_training, name=scope)
            self.conv_ados6_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

        # conv6_2
        with tf.compat.v1.name_scope("conv_ados6_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 128, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados6_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados6_2 = self.batch_normalization(conv, 128, is_training, name=scope)
            self.parameters += [kernel]

        self.conv_ados_6 = tf.compat.v1.nn.relu(self.conv_ados_5 + self.conv_ados6_2, name=scope)

        # block 4
        # conv7_1
        with tf.compat.v1.name_scope("conv_ados7_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 128, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_6, kernel, [1, 4, 4, 1], padding="SAME")
            out = self.batch_normalization(conv, 128, is_training, name=scope)
            self.conv_ados7_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

        # conv7_2
        with tf.compat.v1.name_scope("conv_ados7_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 128, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados7_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados7_2 = self.batch_normalization(conv, 256, is_training, name=scope)
            self.parameters += [kernel]

        # conv7_3 残差模块
        with tf.compat.v1.name_scope("conv_ados7_3") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 1, 128, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_6, kernel, [1, 4, 4, 1], padding="SAME")
            self.conv_ados7_3 = self.batch_normalization(conv, 256, is_training, name=scope)
            self.parameters += [kernel]

        self.conv_ados_7 = tf.compat.v1.nn.relu(self.conv_ados7_2 + self.conv_ados7_3, name=scope)

        # conv8_1
        with tf.compat.v1.name_scope("conv_ados8_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 256, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados_7, kernel, [1, 1, 1, 1], padding="SAME")
            out = self.batch_normalization(conv, 256, is_training, name=scope)
            self.conv_ados8_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel]

        # conv8_2
        with tf.compat.v1.name_scope("conv_ados8_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([1, 9, 256, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv_ados8_1, kernel, [1, 1, 1, 1], padding="SAME")
            self.conv_ados8_2 = self.batch_normalization(conv, 256, is_training, name=scope)
            self.parameters += [kernel]

        self.conv_ados_8 = tf.compat.v1.nn.relu(self.conv_ados_7 + self.conv_ados8_2, name=scope)



        height = int(self.conv_ados_8.get_shape().as_list()[1])  # =1
        weight = int(self.conv_ados_8.get_shape().as_list()[2])  # =50176/4/4/4/4/4=149
        print(height,weight)
        # pool2
        # 全局平均池化
        self.audio_pool2 = tf.compat.v1.nn.avg_pool(
            self.conv_ados_8,
            ksize=[1, height, weight, 1],
            strides=[1, height, weight, 1],
            padding="VALID",
            name="audio_pool2",
        )

    def resnet_part(self):

        # fc1
        with tf.compat.v1.name_scope("reg_head") as scope:
            # shape = 256
            shape = int(np.prod(self.audio_pool2.get_shape()[1:]))+int(np.prod(self.pool2.get_shape()[1:]))

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

            self.pool2_flat = tf.compat.v1.nn.l2_normalize(
                tf.compat.v1.reshape(self.pool2, [-1, int(shape/2)]), 1
            )
            self.audio_pool2_flat = tf.compat.v1.nn.l2_normalize(
                tf.compat.v1.reshape(self.audio_pool2, [-1, int(shape/2)]), 1
            )

            # 连接 512-d
            self.concat = tf.compat.v1.concat([self.pool2_flat, self.audio_pool2_flat], 1)

            # 添加偏置项 tk.nn.bias_add()
            # 函数：tf.matmul 表示：将矩阵 a 乘以矩阵 b,生成a * b
            self.reg_head = tf.compat.v1.nn.bias_add(
                tf.compat.v1.matmul(self.concat, fc1w), fc1b, name=scope
            )
            self.parameters += [fc1w, fc1b]


    def load_trained_model(self, pickle_file, sess):
        with open(pickle_file, "rb") as pfile:
            param = pickle.load(pfile)
        for i in range(len(param)):
            sess.run(self.parameters[i].assign(param[i]))

