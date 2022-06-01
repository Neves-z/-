import warnings
import os

import numpy as np
import tensorflow as tf
from Resnet import Resnet_18

tf.compat.v1.disable_eager_execution()
warnings.filterwarnings("ignore")

### 只能调试 运行出错

BATCH_SIZE = 10
REG_PENALTY = 0
N_EPOCHS = 2

NUM_VAL_VIDEOS = 4000  # 2000

def test_resnet_val():
    imgs = tf.compat.v1.placeholder("float", [None, 224, 224, 3], name="image_placeholder")
    ados = tf.compat.v1.placeholder("float", [None, 1, 50176, 1], name="audio_placeholder")

    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.compat.v1.Session(config=config) as sess:
        model = Resnet_18(imgs, ados, REG_PENALTY=REG_PENALTY, is_training=False)
        saver = tf.compat.v1.train.Saver()

        val_filename_queue = tf.compat.v1.train.string_input_producer(
            ["Res_val.tfrecords"], num_epochs=N_EPOCHS
        )
        val_reader = tf.compat.v1.TFRecordReader()
        _, val_serialized_example = val_reader.read(val_filename_queue)
        # Decode the record read by the reader
        val_feature = {
            "val/image": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
            "val/audio": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
            "val/label": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
        }
        val_features = tf.compat.v1.parse_single_example(val_serialized_example, features=val_feature)
        # Convert the image data from string back to the numbers
        val_image = tf.compat.v1.decode_raw(val_features["val/image"], tf.compat.v1.uint8)
        val_audio = tf.compat.v1.decode_raw(val_features["val/audio"], tf.compat.v1.float32)
        val_label = tf.compat.v1.decode_raw(val_features["val/label"], tf.compat.v1.float32)

        # Reshape image data into the original shape
        val_image = tf.compat.v1.reshape(val_image, [224, 224, 3])
        val_audio = tf.compat.v1.reshape(val_audio, [1, 50176, 1])
        val_label = tf.compat.v1.reshape(val_label, [5])
        val_images, val_audios, val_labels = tf.compat.v1.train.shuffle_batch(
            [val_image, val_audio, val_label],
            batch_size=BATCH_SIZE,
            capacity=200,
            min_after_dequeue=BATCH_SIZE,
            allow_smaller_final_batch=True,
        )
        # 保存模型对象saver

        '''
        if os.path.exists('model_res/checkpoint'):  # 判断模型是否存在
            print("yes!")
            saver.restore(sess, "model_res/model_resnet.ckpt")  # 存在就从模型中恢复变量
        else:
            print("模型不存在!")
            init = tf.compat.v1.global_variables_initializer()  # 不存在就初始化变量
            sess.run(init)
        '''
        # 定义初始化模型所有参数的操作 但需要靠 sess.run 真正执行
        init_op = tf.compat.v1.group(
            tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()
        )
        # 真正初始化全局和局部参数
        sess.run(init_op)


        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord)

        file_list = ["param/param_resnet.pkl"]
        for pickle_file in file_list:
            model.load_trained_model(pickle_file, sess)
            val_acc_list = []
            val_mean_acc_list = []
            sess.run(tf.compat.v1.local_variables_initializer())
            print("Computing Validation Accuracy..")
            error =0
            i = 0
            while i < NUM_VAL_VIDEOS:
                i += BATCH_SIZE
                try:
                    epoch_x, epoch_y, epoch_z = sess.run([val_images, val_audios, val_labels])
                except:
                    print(error, ": Error in reading this batch", i)
                    error += 1
                    if error > 10:
                        break
                    continue
                output = sess.run(
                    [model.output], feed_dict={imgs: epoch_x.astype(np.float32), ados: epoch_y.astype(np.float32)}
                )

                val_acc = 1 - np.absolute(output - epoch_z)
                val_mean_acc = np.mean(val_acc)
                val_acc_list.append(val_acc[0])
                val_mean_acc_list.append(val_mean_acc)

            val_mean_acc = np.mean(val_mean_acc_list)
            val_acc = np.mean(np.concatenate(val_acc_list), axis=0)
            print("Val. Mean Acc:" + str(round(val_mean_acc, 4)))
            print("Val. Acc:", val_acc)
            print("-----------------------------------------------")

        coord.request_stop()
        coord.join(threads)

test_resnet_val()

