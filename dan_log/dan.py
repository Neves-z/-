########################################################################################
#                                                                                      #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import pickle
import warnings

import numpy as np
import tensorflow as tf
from scipy.io import loadmat

warnings.filterwarnings("ignore")


class DAN:
    def __init__(self, imgs, REG_PENALTY=0, preprocess=None):
        self.imgs = imgs
        if preprocess == "vggface":
            self.mean = [123.68, 116.779, 103.939]   #？？[129.1862793, 104.76238251, 93.59396362]
        self.convlayers()
        self.dan_part()
        # sigmoid()
        self.output = tf.compat.v1.nn.sigmoid(self.reg_head, name="output")
        self.cost_reg = REG_PENALTY * tf.compat.v1.reduce_mean(tf.compat.v1.square(self.parameters[-2])) / 2

    def convlayers(self):
        self.parameters = []

        # zero-mean input 零均值化/中心化
        with tf.compat.v1.name_scope("preprocess") as scope:
            # tf.compat.v1.constant:创建一个常数张量  shape 常量尺寸
            mean = tf.compat.v1.constant(
                self.mean, dtype=tf.compat.v1.float32, shape=[1, 1, 1, 3], name="img_mean"
            )
            images = self.imgs - mean

        # conv1_1
        with tf.compat.v1.name_scope("conv1_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 3, 64], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(images, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.compat.v1.Variable(
                tf.compat.v1.constant(0.0, shape=[64], dtype=tf.compat.v1.float32),
                trainable=True,
                name="biases",
            )
            out = tf.compat.v1.nn.bias_add(conv, biases)
            self.conv1_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.compat.v1.name_scope("conv1_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 64, 64], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.compat.v1.Variable(
                tf.compat.v1.constant(0.0, shape=[64], dtype=tf.compat.v1.float32),
                trainable=True,
                name="biases",
            )
            out = tf.compat.v1.nn.bias_add(conv, biases)
            self.conv1_2 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.compat.v1.nn.max_pool(
            self.conv1_2,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME",
            name="pool1",
        )

        # conv2_1
        with tf.compat.v1.name_scope("conv2_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 64, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.compat.v1.Variable(
                tf.compat.v1.constant(0.0, shape=[128], dtype=tf.compat.v1.float32),
                trainable=True,
                name="biases",
            )
            out = tf.compat.v1.nn.bias_add(conv, biases)
            self.conv2_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.compat.v1.name_scope("conv2_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 128, 128], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.compat.v1.Variable(
                tf.compat.v1.constant(0.0, shape=[128], dtype=tf.compat.v1.float32),
                trainable=True,
                name="biases",
            )
            out = tf.compat.v1.nn.bias_add(conv, biases)
            self.conv2_2 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.compat.v1.nn.max_pool(
            self.conv2_2,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME",
            name="pool2",
        )

        # conv3_1
        with tf.compat.v1.name_scope("conv3_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 128, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.compat.v1.Variable(
                tf.compat.v1.constant(0.0, shape=[256], dtype=tf.compat.v1.float32),
                trainable=True,
                name="biases",
            )
            out = tf.compat.v1.nn.bias_add(conv, biases)
            self.conv3_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.compat.v1.name_scope("conv3_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 256, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.compat.v1.Variable(
                tf.compat.v1.constant(0.0, shape=[256], dtype=tf.compat.v1.float32),
                trainable=True,
                name="biases",
            )
            out = tf.compat.v1.nn.bias_add(conv, biases)
            self.conv3_2 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.compat.v1.name_scope("conv3_3") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 256, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.compat.v1.Variable(
                tf.compat.v1.constant(0.0, shape=[256], dtype=tf.compat.v1.float32),
                trainable=True,
                name="biases",
            )
            out = tf.compat.v1.nn.bias_add(conv, biases)
            self.conv3_3 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.compat.v1.nn.max_pool(
            self.conv3_3,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME",
            name="pool3",
        )

        # conv4_1
        with tf.compat.v1.name_scope("conv4_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 256, 512], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.compat.v1.Variable(
                tf.compat.v1.constant(0.0, shape=[512], dtype=tf.compat.v1.float32),
                trainable=True,
                name="biases",
            )
            out = tf.compat.v1.nn.bias_add(conv, biases)
            self.conv4_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.compat.v1.name_scope("conv4_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 512, 512], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.compat.v1.Variable(
                tf.compat.v1.constant(0.0, shape=[512], dtype=tf.compat.v1.float32),
                trainable=True,
                name="biases",
            )
            out = tf.compat.v1.nn.bias_add(conv, biases)
            self.conv4_2 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.compat.v1.name_scope("conv4_3") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 512, 512], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.compat.v1.Variable(
                tf.compat.v1.constant(0.0, shape=[512], dtype=tf.compat.v1.float32),
                trainable=True,
                name="biases",
            )
            out = tf.compat.v1.nn.bias_add(conv, biases)
            self.conv4_3 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.compat.v1.nn.max_pool(
            self.conv4_3,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME",
            name="pool4",
        )

        # conv5_1
        with tf.compat.v1.name_scope("conv5_1") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 512, 512], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.compat.v1.Variable(
                tf.compat.v1.constant(0.0, shape=[512], dtype=tf.compat.v1.float32),
                trainable=True,
                name="biases",
            )
            out = tf.compat.v1.nn.bias_add(conv, biases)
            self.conv5_1 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.compat.v1.name_scope("conv5_2") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 512, 512], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.compat.v1.Variable(
                tf.compat.v1.constant(0.0, shape=[512], dtype=tf.compat.v1.float32),
                trainable=True,
                name="biases",
            )
            out = tf.compat.v1.nn.bias_add(conv, biases)
            self.conv5_2 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.compat.v1.name_scope("conv5_3") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 512, 512], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.compat.v1.Variable(
                tf.compat.v1.constant(0.0, shape=[512], dtype=tf.compat.v1.float32),
                trainable=True,
                name="biases",
            )
            out = tf.compat.v1.nn.bias_add(conv, biases)
            self.conv5_3 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.compat.v1.nn.max_pool(
            self.conv5_3,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME",
            name="pool5",
        )
        # MaxPool6
        #self.pool5:7*7*512
        self.maxpool5 = tf.compat.v1.nn.max_pool(
            self.pool5,
            ksize=[1, 7, 7, 1],
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="maxpool5",
        )
        # self.maxpool5:512-d
        # AvgPool6
        self.avgpool5 = tf.compat.v1.nn.avg_pool(
            self.pool5,
            ksize=[1, 7, 7, 1],
            strides=[1, 7, 7, 1],
            padding="SAME",
            name="avgpool5",
        )

    def dan_part(self):

        # fc1
        with tf.compat.v1.name_scope("reg_head") as scope:

            #tensorflow查看形状XXX.get_shape()
            # np.prod 计算所谓元素的乘积  np.prod([1,2,3])=6
            shape = 2 * int(np.prod(self.maxpool5.get_shape()[1:]))
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

            maxpool5_flat = tf.compat.v1.nn.l2_normalize(
                tf.compat.v1.reshape(self.maxpool5, [-1, int(shape / 2)]), 1
            )
            avgpool5_flat = tf.compat.v1.nn.l2_normalize(
                tf.compat.v1.reshape(self.avgpool5, [-1, int(shape / 2)]), 1
            )
            # 连接 1024-d
            self.concat = tf.compat.v1.concat([maxpool5_flat, avgpool5_flat], 1)
            # 添加偏置项 tk.nn.bias_add()
            # 函数：tf.matmul 表示：将矩阵 a 乘以矩阵 b,生成a * b
            self.reg_head = tf.compat.v1.nn.bias_add(
                tf.compat.v1.matmul(self.concat, fc1w), fc1b, name=scope
            )
            self.parameters += [fc1w, fc1b]

    def initialize_with_vggface(self, weight_file, sess):
        data = loadmat(weight_file)
        layers = data["layers"][0]
        i = 0
        for layer in layers:
            name = layer[0]["name"][0][0]
            layer_type = layer[0]["type"][0][0]
            if layer_type == "conv" and name[0:2] != "fc":
                kernel, bias = layer[0]["weights"][0][0]
                sess.run(self.parameters[i].assign(kernel))
                sess.run(self.parameters[i + 1].assign(bias.reshape(bias.shape[0])))
                i += 2

    def load_trained_model(self, pickle_file, sess):
        with open(pickle_file, "rb") as pfile:
            param = pickle.load(pfile)
        for i in range(len(param)):
            sess.run(self.parameters[i].assign(param[i]))
