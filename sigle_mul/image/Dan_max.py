import pickle
import warnings

import numpy as np
import tensorflow as tf
from scipy.io import loadmat
warnings.filterwarnings("ignore")


class image_net:
    def __init__(self, imgs, REG_PENALTY=0):
        self.imgs = imgs
        self.parameters = []
        self.mean = [123.68, 116.779, 103.939]
        self.convlayers()
        self.net_part()
        self.output = tf.compat.v1.nn.sigmoid(self.reg_head, name="output")

        # self.parameters[-2] = 获取列表倒数第二个元素 即全连接层weights
        self.cost_reg = REG_PENALTY * tf.compat.v1.reduce_mean(tf.compat.v1.square(self.parameters[-2])) / 2


    def convlayers(self):

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

        # conv3_4
        with tf.compat.v1.name_scope("conv3_4") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 256, 256], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv3_3, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.compat.v1.Variable(
                tf.compat.v1.constant(0.0, shape=[256], dtype=tf.compat.v1.float32),
                trainable=True,
                name="biases",
            )
            out = tf.compat.v1.nn.bias_add(conv, biases)
            self.conv3_4 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.compat.v1.nn.max_pool(
            self.conv3_4,
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

        # conv4_4
        with tf.compat.v1.name_scope("conv4_3") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 512, 512], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv4_3, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.compat.v1.Variable(
                tf.compat.v1.constant(0.0, shape=[512], dtype=tf.compat.v1.float32),
                trainable=True,
                name="biases",
            )
            out = tf.compat.v1.nn.bias_add(conv, biases)
            self.conv4_4 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.compat.v1.nn.max_pool(
            self.conv4_4,
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

        # MaxPool5_2
        self.maxpool5_2 = tf.nn.max_pool(
            self.conv5_2,
            ksize=[1, 14, 14, 1],
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="maxpool5_2",
        )

        # AvgPool5_2
        self.avgpool5_2 = tf.nn.avg_pool(
            self.conv5_2,
            ksize=[1, 14, 14, 1],
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="avgpool5_2",
        )


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

        # MaxPool5_3
        self.maxpool5_3 = tf.nn.max_pool(
            self.conv5_3,
            ksize=[1, 14, 14, 1],
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="maxpool5_3",
        )

        # AvgPool5_3
        self.avgpool5_3 = tf.nn.avg_pool(
            self.conv5_3,
            ksize=[1, 14, 14, 1],
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="avgpool5_3",
        )


        # conv5_4
        with tf.compat.v1.name_scope("conv5_3") as scope:
            kernel = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([3, 3, 512, 512], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.compat.v1.nn.conv2d(self.conv5_3, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.compat.v1.Variable(
                tf.compat.v1.constant(0.0, shape=[512], dtype=tf.compat.v1.float32),
                trainable=True,
                name="biases",
            )
            out = tf.compat.v1.nn.bias_add(conv, biases)
            self.conv5_4 = tf.compat.v1.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]


        # pool5
        self.pool5 = tf.compat.v1.nn.max_pool(
            self.conv5_4,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME",
            name="pool5",
        )
        # MaxPool6
        #self.pool5:7*7*512
        self.maxpool5_4 = tf.compat.v1.nn.max_pool(
            self.pool5,
            ksize=[1, 7, 7, 1],
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="maxpool5_4",
        )
        # self.maxpool5:512-d
        # AvgPool6
        self.avgpool5_4 = tf.compat.v1.nn.avg_pool(
            self.pool5,
            ksize=[1, 7, 7, 1],
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="avgpool5_4",
        )


    def net_part(self):
        # fc1
        with tf.compat.v1.name_scope("reg_head") as scope:

            shape = 6 * int(np.prod(self.maxpool5_4.get_shape()[1:]))   # 512*6 = 3072

            fc1w = tf.compat.v1.Variable(
                tf.compat.v1.truncated_normal([shape, 5], dtype=tf.compat.v1.float32, stddev=1e-1),
                name="weights",
            )
            fc1b = tf.compat.v1.Variable(
                tf.compat.v1.constant(1.0, shape=[5], dtype=tf.compat.v1.float32),
                trainable=True,
                name="biases",
            )

            self.maxpool5_2_flat = tf.compat.v1.nn.l2_normalize(
                tf.compat.v1.reshape(self.maxpool5_2, [-1, int(shape / 6)]), 1
            )
            self.avgpool5_2_flat = tf.compat.v1.nn.l2_normalize(
                tf.compat.v1.reshape(self.avgpool5_2, [-1, int(shape / 6)]), 1
            )
            self.maxpool5_3_flat = tf.compat.v1.nn.l2_normalize(
                tf.compat.v1.reshape(self.maxpool5_3, [-1, int(shape / 6)]), 1
            )
            self.avgpool5_3_flat = tf.compat.v1.nn.l2_normalize(
                tf.compat.v1.reshape(self.avgpool5_3, [-1, int(shape / 6)]), 1
            )
            self.maxpool5_4_flat = tf.compat.v1.nn.l2_normalize(
                tf.compat.v1.reshape(self.maxpool5_4, [-1, int(shape / 6)]), 1
            )
            self.avgpool5_4_flat = tf.compat.v1.nn.l2_normalize(
                tf.compat.v1.reshape(self.avgpool5_4, [-1, int(shape / 6)]), 1
            )


            # 连接 3072-d
            self.concat = tf.compat.v1.concat([self.maxpool5_2_flat, self.avgpool5_2_flat, self.maxpool5_3_flat,
                                               self.avgpool5_3_flat,self.maxpool5_4_flat, self.avgpool5_4_flat], 1)

            self.reg_head = tf.compat.v1.nn.bias_add(
                tf.compat.v1.matmul(self.concat, fc1w), fc1b, name=scope
            )
            self.parameters += [fc1w, fc1b]

    def initialize_with_vgg_19(self, weight_file, sess):
        data = loadmat(weight_file)
        layers = data["layers"][0]
        i = 0
        for layer in layers:
            name = layer[0]["name"][0][0]
            layer_type = layer[0]["type"][0][0]
            if layer_type == "conv" and name[0:2] != "fc":
                kernel, bias = layer[0]["weights"][0][0]
                sess.run(self.parameters[i].assign(kernel))
                sess.run(self.parameters[i + 1].assign(bias.reshape(bias.shape[1])))
                i += 2

    def load_trained_model(self, pickle_file, sess):
        with open(pickle_file, "rb") as pfile:
            param = pickle.load(pfile)
        for i in range(len(param)):
            sess.run(self.parameters[i].assign(param[i]))



