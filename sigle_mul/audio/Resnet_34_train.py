import pickle
import time
import os
import warnings

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from Resnet_34 import audio_net
from remtime import *

warnings.filterwarnings("ignore")

LEARNING_RATES = [0.01, 0.005, 0.001, 0.0005, 0.0001]  # 学习率
BOUNDARIES = [5000, 12000, 20000, 30000]  # 学习衰减轮数   NUM_TRAIN_VIDEOS*N_EPOCHS/BATCH_SIZE=9000*4=36000

BATCH_SIZE = 32
N_EPOCHS = 4
REG_PENALTY = 0.1
NUM_AUDIOS = 288000
NUM_TEST_AUDIOS = 64000



ados = tf.compat.v1.placeholder("float", [None, 1, 50176, 1], name="audio_placeholder")
values = tf.compat.v1.placeholder("float", [None, 5], name="value_placeholder")

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)




with tf.compat.v1.Session(config=config) as sess:

    model = audio_net(ados, REG_PENALTY=REG_PENALTY, is_training=True)
    output = model.output

    # 初始化轮数计数器，定义为不可训练
    global_step = tf.Variable(0, trainable=False)
    # 定义分段常数衰减的学习率
    learning_rate = tf.compat.v1.train.piecewise_constant(global_step, boundaries=BOUNDARIES, values=LEARNING_RATES)

    cost = tf.compat.v1.reduce_mean(tf.compat.v1.squared_difference(model.output, values)) + model.cost_reg
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

    tr_filename_queue = tf.compat.v1.train.string_input_producer(
        ['Res34_train.tfrecords'], num_epochs=2 * N_EPOCHS
    )

    tr_reader = tf.compat.v1.TFRecordReader()
    _, tr_serialized_example = tr_reader.read(tr_filename_queue)  # 返回文件名和文件
    # Decode the record read by the reader
    # 定义如何解析数据 FixedLenFeature用于处理定长的特征
    tr_feature = {
        "train/audio": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
        "train/label": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
    }
    # 将Example协议缓冲区(protocol buffer)解析为张量
    tr_features = tf.compat.v1.parse_single_example(tr_serialized_example, features=tr_feature)
    # Convert the image data from string back to the numbers   修改数据类型，获取图片数据
    tr_audio = tf.compat.v1.decode_raw(tr_features["train/audio"], tf.compat.v1.float32)
    tr_label = tf.compat.v1.decode_raw(tr_features["train/label"], tf.compat.v1.float32)

    tr_audio = tf.compat.v1.reshape(tr_audio, [1, 50176, 1])
    tr_label = tf.compat.v1.reshape(tr_label, [5])

    tr_audios, tr_labels = tf.compat.v1.train.shuffle_batch(
        [tr_audio, tr_label],
        batch_size=BATCH_SIZE,
        capacity=200,
        min_after_dequeue=BATCH_SIZE,
        allow_smaller_final_batch=True,
    )

    val_filename_queue = tf.compat.v1.train.string_input_producer(
        ["Res34_val.tfrecords"], num_epochs=N_EPOCHS
    )
    val_reader = tf.compat.v1.TFRecordReader()
    _, val_serialized_example = val_reader.read(val_filename_queue)
    # Decode the record read by the reader
    val_feature = {
        "val/audio": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
        "val/label": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
    }
    val_features = tf.compat.v1.parse_single_example(val_serialized_example, features=val_feature)
    # Convert the image data from string back to the numbers
    val_audio = tf.compat.v1.decode_raw(val_features["val/audio"], tf.compat.v1.float32)
    val_label = tf.compat.v1.decode_raw(val_features["val/label"], tf.compat.v1.float32)

    # Reshape image data into the original shape
    val_audio = tf.compat.v1.reshape(val_audio, [1, 50176, 1])
    val_label = tf.compat.v1.reshape(val_label, [5])
    val_audios, val_labels = tf.compat.v1.train.shuffle_batch(
        [val_audio, val_label],
        batch_size=BATCH_SIZE,
        capacity=200,
        min_after_dequeue=BATCH_SIZE,
        allow_smaller_final_batch=True,
    )
    # 定义初始化模型所有参数的操作 但需要靠 sess.run 真正执行
    init_op = tf.compat.v1.group(
        tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()
    )
    # 真正初始化全局和局部参数
    sess.run(init_op)

    coord = tf.compat.v1.train.Coordinator()
    threads = tf.compat.v1.train.start_queue_runners(coord=coord)

    loss_list = []
    param_ados_num = 1
    try:
        os.makedirs("param")
    except OSError:
        print("Error: Creating param")

    for epoch in range(N_EPOCHS):
        tr_acc_list = []
        tr_mean_acc_list = []
        val_acc_list = []
        val_mean_acc_list = []
        sess.run(tf.compat.v1.local_variables_initializer())
        i = 0
        error = 0
        stime = time.time()
        print("------------------ Epoch", epoch + 1, "--------------------")
        while i < NUM_AUDIOS:
            i += BATCH_SIZE
            # 每次处理BATCH_SIZE个数据
            try:
                # 图片数据列表，对应标签，一次批处理BATCH_SIZE（25)个
                epoch_x, epoch_y = sess.run([tr_audios, tr_labels])
            except:
                print(error, ": Error in reading this batch:", int(i/BATCH_SIZE))
                error += 1
                if error > 10:
                    break
                continue
            #feed_dict的作用是给使用placeholder创建出来的tensor赋值
            _, c = sess.run(
                [optimizer, cost],
                feed_dict={ados: epoch_x.astype(np.float32), values: epoch_y},
            )
            loss_list.append(np.power(c, 0.5))  #求loss时平方了 所以c^0.5

            if not i % 3200: #2000
                # 训练完成的图片的百分比
                per = float(i) / NUM_AUDIOS * 100
                print(
                    "Epoch:"
                    + str(round(per, 2))      # 保留两位小数
                    + "% Of "
                    + str(epoch + 1)
                    + "/"
                    + str(N_EPOCHS)
                    + ", Batch loss:"
                    + str(round(c, 4))
                )
                ftime = time.time()
                # 计算预计训练完的剩余时间
                remtime = (ftime - stime) * ((NUM_AUDIOS - i) / 3200)
                stime = ftime
                # 转化为时分秒
                printTime(remtime)


        # 一次epoch结束，把训练完的参数保存
        with open("param/param_res34_ados" + str(param_ados_num) + ".pkl", "wb") as pfile:
            pickle.dump(sess.run(model.ado_parameters), pfile, pickle.HIGHEST_PROTOCOL)
            print(str(param_ados_num) + " weights Saved!!")
            param_ados_num += 1


        print("------------------ Epoch", epoch + 1, "--------------------")
        sess.run(tf.compat.v1.local_variables_initializer())
        print("Computing Training Accuracy..")
        i = 0
        error = 0
        while i < NUM_AUDIOS:
            i += BATCH_SIZE
            try:
                epoch_x, epoch_y = sess.run([tr_audios, tr_labels])
            except:
                print(error,"Error in reading this batch:",int(i/BATCH_SIZE))
                error += 1
                if error > 10:
                    break
                continue
            output = sess.run(
                [model.output], feed_dict={ados: epoch_x.astype(np.float32)}
            )
            tr_acc = 1 - np.absolute(output - epoch_y)
            tr_mean_acc = np.mean(tr_acc)
            tr_acc_list.append(tr_acc[0])
            tr_mean_acc_list.append(tr_mean_acc)

        tr_acc = np.mean(np.concatenate(tr_acc_list), axis=0)
        tr_mean_acc = np.mean(tr_mean_acc_list)
        print("Tr. Acc:" + str(tr_acc))
        print("Tr. Mean Acc:" + str(round(tr_mean_acc, 4)))
        print("Computing Validation Accuracy..")
        i = 0
        error = 0
        while i < NUM_TEST_AUDIOS:
            i += BATCH_SIZE
            try:
                epoch_x, epoch_y = sess.run([val_audios, val_labels])
            except:
                print(error, "Error in reading this batch:", int(i/BATCH_SIZE))
                error += 1
                if error > 10:
                    break
                continue
            output = sess.run(
                [model.output], feed_dict={ados: epoch_x.astype(np.float32)}
            )
            val_acc = 1 - np.absolute(output - epoch_y)
            val_mean_acc = np.mean(val_acc)
            val_acc_list.append(val_acc[0])
            val_mean_acc_list.append(val_mean_acc)

        val_acc = np.mean(np.concatenate(val_acc_list),axis=0)
        val_mean_acc = np.mean(val_mean_acc_list)
        print("Val. Acc:" + str(val_acc))
        print("Val. Mean Acc:" + str(round(val_mean_acc, 4)))
        print("Epoch_" + str(epoch + 1) + " completed out of " + str(N_EPOCHS))
        print("-----------------------------------------------")

    coord.request_stop()
    # Wait for threads to stop
    coord.join(threads)

    #训练完成后的变量保存
    saver = tf.compat.v1.train.Saver()
    saver.save(sess, "model_res34_ado/model_audio.ckpt")
    print("Session Saved!!")
    with open("loss_res34_ado.pkl", "wb") as pfile:
        pickle.dump(loss_list, pfile, pickle.HIGHEST_PROTOCOL)
    print("Loss_res34_ado List Saved!!")
