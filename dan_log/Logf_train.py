import pickle
import time
import warnings

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from Logf_reg import Audio_reg
from remtime import *

warnings.filterwarnings("ignore")


LEARNING_RATE = 0.0005
BATCH_SIZE = 10
N_EPOCHS = 10
REG_PENALTY = 0
NUM_AUDIOS = 6000 # 6000
NUM_TEST_AUDIOS = 2000 #2000


# placeholder为将始终输入的张量插入占位符。
# 它的值必须使用feed_dict可选参数输入到Session.run（）
Ados = tf.compat.v1.placeholder("float", [None,1,79534], name="audio_placeholder")
values = tf.compat.v1.placeholder("float", [None, 5], name="value_placeholder")

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
# log_device_placement=True : 是否打印设备分配日志
# allow_soft_placement=True如果你指定的设备不存在，允许TF自动分配设备
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

# 让参数设置生效的方法
# session = tf.Session(config=config)
with tf.compat.v1.Session(config=config) as sess:

    model = Audio_reg(Ados, REG_PENALTY=REG_PENALTY)
    output = model.output
    # 计算损失
    # reduce_mean为平均值 即批处理25个数据的损失的平均值
    cost = tf.compat.v1.reduce_mean(tf.compat.v1.squared_difference(model.output, values)) + model.cost_reg
    # Adam优化算法是一种对随机梯度下降法的扩展，最近在计算机视觉和自然语言处理中广泛应用于深度学习应用。
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)


    # 根据文件名生成一个解析队列  每次从一个tensor列表（训练集）中按顺序或者随机抽取出一个tensor放入文件名队列。
    # num_epochs: 可选参数，是一个整数值，代表迭代的次数，
    # 如果设置 num_epochs=None,生成器可以无限次遍历tensor列表，如果设置为 num_epochs=N，生成器只能遍历tensor列表N次。
    tr_filename_queue = tf.compat.v1.train.string_input_producer(
        ["Logf_audio_train.tfrecords"], num_epochs=2*N_EPOCHS
    )

    tr_reader = tf.compat.v1.TFRecordReader()
    _, tr_serialized_example = tr_reader.read(tr_filename_queue)   #返回文件名和文件
    # Decode the record read by the reader
    # 定义如何解析数据 FixedLenFeature用于处理定长的特征
    tr_feature = {
        "train/audio": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
        "train/label": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
    }
    # 将Example协议缓冲区(protocol buffer)解析为张量
    tr_features = tf.compat.v1.parse_single_example(tr_serialized_example, features=tr_feature)
    # Convert the image data from string back to the numbers  解压
    tr_audio = tf.compat.v1.decode_raw(tr_features["train/audio"], tf.compat.v1.float64)
    tr_label = tf.compat.v1.decode_raw(tr_features["train/label"], tf.compat.v1.float32)
    # Reshape image data into the original shape
    tr_audio = tf.compat.v1.reshape(tr_audio, [1,79534])
    tr_label = tf.compat.v1.reshape(tr_label, [5])
    # 批处理队列 使用shuffle_batch可以随机打乱数据
    tr_audios, tr_labels = tf.compat.v1.train.shuffle_batch(
        [tr_audio, tr_label],
        batch_size=BATCH_SIZE,
        capacity=200,
        min_after_dequeue=BATCH_SIZE,
        allow_smaller_final_batch=True,
    )
    val_filename_queue = tf.compat.v1.train.string_input_producer(
        ["Logf_audio_val.tfrecords"], num_epochs=N_EPOCHS
    )
    val_reader = tf.compat.v1.TFRecordReader()
    _, val_serialized_example = val_reader.read(val_filename_queue)
    # Decode the record read by the reader
    val_feature = {
        "val/audio": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
        "val/label": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
    }
    val_features = tf.compat.v1.parse_single_example(val_serialized_example, features=val_feature)
    val_audio = tf.compat.v1.decode_raw(val_features["val/audio"], tf.compat.v1.float64)
    val_label = tf.compat.v1.decode_raw(val_features["val/label"], tf.compat.v1.float32)
    # Reshape image data into the original shape
    val_audio = tf.compat.v1.reshape(val_audio, [1, 79534])
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


    # 协调器 tf.train.Coordinator
    # 可以用来同时停止多个工作线程并且向那个在等待所有工作线程终止的程序报告异常
    coord = tf.compat.v1.train.Coordinator()
    # 入队线程启动器 tf.train.start_queue_runners
    # 用来启动多个工作线程同时将多个tensor（训练数据）推送入文件名称队列Filename Queue中
    threads = tf.compat.v1.train.start_queue_runners(coord=coord)


    loss_list = []
    param_num = 1

    for epoch in range(N_EPOCHS):
        tr_acc_list = []
        tr_mean_acc_list = []
        val_acc_list = []
        val_mean_acc_list = []
        sess.run(tf.compat.v1.local_variables_initializer())
        i = 0
        error = 0
        stime = time.time()
        print("------------------ Epoch", epoch+1, "--------------------")
        while i < NUM_AUDIOS:

            i += BATCH_SIZE
            # 每次处理BATCH_SIZE个数据
            try:
                # 图片数据列表，对应标签，一次批处理BATCH_SIZE（25)个
                epoch_x, epoch_y = sess.run([tr_audios, tr_labels])
            except:
                print(error, ": Error in reading this batch")
                error += 1
                if error > 10:
                    break
                continue
            #feed_dict的作用是给使用placeholder创建出来的tensor赋值
            _, c = sess.run(
                [optimizer, cost],
                feed_dict={Ados: epoch_x.astype(np.float64), values: epoch_y},
            )
            loss_list.append(np.power(c, 0.5))  #求loss时平方了 所以c^0.5
            # 每训练完200个音频就打印一次进度
            if not i % 3000: #2000
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
                remtime = (ftime - stime) * ((NUM_AUDIOS - i) / 1000)
                # remtime = (ftime - stime) * ((NUM_IMAGES - i) / (NUM_IMAGES / x))
                stime = ftime
                # 转化为时分秒
                printTime(remtime)
            '''
            # 每1000个音频保存一下参数
            if not i % 3200:  #20000
                with open("param/param_dl_ados" + str(param_num) + ".pkl", "wb") as pfile:
                    pickle.dump(
                        sess.run(model.parameters), pfile, pickle.HIGHEST_PROTOCOL
                    )
                print(str(param_num) + " weights Saved!!")
                param_num += 1
            '''
        # 一次epoch结束，把训练完的参数保存
        with open("param/param_dl_ados" + str(param_num) + ".pkl", "wb") as pfile:
            pickle.dump(sess.run(model.parameters), pfile, pickle.HIGHEST_PROTOCOL)
            print(str(param_num) + " weights Saved!!")
            param_num += 1
        sess.run(tf.compat.v1.local_variables_initializer())
        print("------------------ Epoch",epoch+1,"--------------------")
        print("Computing Training Accuracy..")
        i = 0
        error = 0
        while i < NUM_AUDIOS:
            i += BATCH_SIZE
            try:
                epoch_x, epoch_y = sess.run([tr_audios, tr_labels])
            except:
                print("Error in reading this batch")
                error += 1
                if error > 10:
                    break
                continue
            output = sess.run(
                [model.output], feed_dict={Ados: epoch_x.astype(np.float64)}
            )
            tr_acc = 1 - np.absolute(output - epoch_y)
            tr_mean_acc = np.mean(tr_acc )
            tr_acc_list.append(tr_acc[0])
            tr_mean_acc_list.append(tr_mean_acc)
        tr_acc = np.mean(np.concatenate(tr_acc_list), axis=0)
        tr_mean_acc = np.mean(tr_mean_acc_list)
        print("Tr.  Acc:" + str(tr_acc))
        print("Tr. Mean Acc:" + str(round(tr_mean_acc, 4)))
        print("Computing Validation Accuracy..")
        sess.run(tf.compat.v1.local_variables_initializer())
        i = 0
        error = 0
        while i < NUM_TEST_AUDIOS:
            i += BATCH_SIZE
            try:
                epoch_x, epoch_y = sess.run([val_audios, val_labels])
            except:
                print("Error in reading this batch")
                error += 1
                if error > 10:
                    break
                continue
            output = sess.run(
                [model.output], feed_dict={Ados: epoch_x.astype(np.float64)}
            )
            val_acc = 1 - np.absolute(output - epoch_y)
            val_mean_acc = np.mean(val_acc)
            val_acc_list.append(val_acc[0])
            val_mean_acc_list.append(val_mean_acc)
        val_acc = np.mean(np.concatenate(val_acc_list), axis=0)
        val_mean_acc = np.mean(val_mean_acc_list)
        print("Val.  Acc:" + str(val_acc))
        print("Val. Mean Acc:" + str(round(val_mean_acc, 4)))
        print("Epoch_" + str(epoch + 1) + " completed out of " + str(N_EPOCHS))
        print("-----------------------------------------------")

    coord.request_stop()
    # Wait for threads to stop
    coord.join(threads)

    #训练完成后的变量保存
    saver = tf.compat.v1.train.Saver()
    saver.save(sess, "model/model_logf/model_audio.ckpt")
    print("Session Saved!!")
    with open("loss/loss_logf_audio.pkl", "wb") as pfile:
        pickle.dump(loss_list, pfile, pickle.HIGHEST_PROTOCOL)
    print("Audio_Loss List Saved!!")
