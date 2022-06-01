import warnings
import os
import sys
import random
import pickle
import pandas as pd
import glob


import numpy as np
import tensorflow as tf
from Resnet import Resnet_18
from PIL import Image
import scipy.io.wavfile as wav

tf.compat.v1.disable_eager_execution()
warnings.filterwarnings("ignore")




def load_pickle(pickle_file):
    with open(pickle_file, "rb") as f:
        pickle_data = pickle.load(f, encoding="latin1")
        df = pd.DataFrame(pickle_data)
        df.reset_index(inplace=True)
        del df["interview"]
        df.columns = [
            "VideoName",
            "ValueExtraversion",
            "ValueNeuroticism",
            "ValueAgreeableness",
            "ValueConscientiousness",
            "ValueOpenness",
        ]
    return df

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_image(addr):
    image = Image.open(addr).resize((224, 224), Image.ANTIALIAS)
    '''
    img = img.transpose(Image.FLIP_LEFT_RIGHT)  #左右对换。
    # img = img.transpose(Image.FLIP_TOP_BOTTOM)  # 上下对换
    '''
    img = np.array(image)
    img = img.astype(np.uint8)
    return img
"""
# 固定采样
def process_wav(wav_file):
    (rate, sig) = wav.read(wav_file)  # 674816
    val_wav = []
    i = 0
    j = 0
    while i < 25088:
        if j < len(sig):
            val_wav.append(sig[j])
            i += 1
            j += 26
        else:
            j -= 26


    a = np.concatenate(val_wav)
    b = a.reshape(1, 50176, 1)
    b = b.astype(np.float32)
    if b.shape != (1, 50176, 1):
        print("wav error!")
    return b
"""

## 音频随机采样
def process_wav(wav_file):
    (rate, sig) = wav.read(wav_file)  # 674816
    val_wav = []
    ## 随机采样
    index = random.sample(range(0, len(sig)), 25088)
    index.sort()
    index = np.array(index)
    for i in range(len(index)):
        val_wav.append(sig[index[i]])

    a = np.concatenate(val_wav)
    b = a.reshape(1, 50176, 1)
    b = b.astype(np.float32)
    if b.shape != (1, 50176, 1):
        print("wav length error!")
    return b

def predict_val(NUM_VAL,PER_VAL):

    BATCH_SIZE = PER_VAL
    TOTAL_VAL = PER_VAL * NUM_VAL
    # 加载训练好的参数
    file_list = ["param/param_resnet1.pkl",
                 "param/param_resnet2.pkl",
                 "param/param_resnet3.pkl",
                 "param/param_resnet4.pkl"]

    imgs = tf.compat.v1.placeholder("float", [None, 224, 224, 3], name="image_placeholder")
    ados = tf.compat.v1.placeholder("float", [None, 1, 50176, 1], name="audio_placeholder")

    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.compat.v1.Session(config=config) as sess:
        model = Resnet_18(imgs, ados, REG_PENALTY=0, is_training=False)

        val_filename_queue = tf.compat.v1.train.string_input_producer(
            ["val_acc/Resnet/val_resnet.tfrecords"], num_epochs=len(file_list)
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
        # 不打乱
        val_images, val_audios, val_labels = tf.compat.v1.train.batch(
            [val_image, val_audio, val_label],
            batch_size=BATCH_SIZE,
            capacity=200,
            allow_smaller_final_batch=True,
        )

        init_op = tf.compat.v1.group(
            tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()
        )
        sess.run(init_op)


        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord)

        print("Length of total val:", TOTAL_VAL)
        print("Num of epoch:", len(file_list))
        for j in range(len(file_list)):
            model.load_trained_model(file_list[j], sess)
            val_acc_list = []
            val_mean_acc_list = []
            sess.run(tf.compat.v1.local_variables_initializer())
            print("-------------------epoch ", j + 1, "------------------")
            print("Computing Validation Accuracy..")
            error = 0
            i = 0

            while i < TOTAL_VAL:
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
                val_acc = np.mean(1 - np.absolute(output[0] - epoch_z), axis=0)
                val_mean_acc = np.mean(val_acc)
                val_acc_list.append(val_acc)
                val_mean_acc_list.append(val_mean_acc)


            if len(val_acc_list) != NUM_VAL or len(val_mean_acc_list) != NUM_VAL:
                print('error')

            val_acc = np.mean(val_acc_list, axis=0)
            val_mean_acc = np.mean(val_mean_acc_list)
            print("val_acc: ", val_acc)
            print("val_mean_acc: ", val_mean_acc)
            print("----------------------------------------------")


        coord.request_stop()
        coord.join(threads)







def val_acc():
    df = load_pickle("Annotations/annotation_validation.pkl")
    NUM_VID = len(df)  # 2000
    NUM_VAL = NUM_VID
    PER_VAL = 32
    image_list = []
    audio_list = []
    labels = []
    for i in range(NUM_VAL):
        index = 1
        for j in range(PER_VAL):

            image_name = glob.glob(
                "ImageData/validationData/"
                + (df["VideoName"].iloc[i]).split(".mp4")[0]
                + "/frame" + str(index) + ".jpg"
            )
            index += 3
            image_list += [image_name[0]]


        audio_name = glob.glob(
            "VoiceData/validationData/"
            + (df["VideoName"].iloc[i]).split(".mp4")[0]
            + ".wav"
        )
        audio_list += [audio_name[0]] * PER_VAL
        labels += [np.array(df.drop(["VideoName"], 1, inplace=False).iloc[i]).astype(np.float32)] * PER_VAL

    if (len(image_list) != PER_VAL * NUM_VAL):
        print("val acc data length error：", len(image_list))
    try:
        os.makedirs("val_acc/Resnet")
    except OSError:
        print("Error: Creating val_acc/Resnet")

    val_filename = "val_acc/Resnet/val_resnet.tfrecords"  # address to save the TFRecords file
    writer = tf.io.TFRecordWriter(val_filename)
    for i in range(len(image_list)):
        img = load_image(image_list[i])
        ado = process_wav(audio_list[i])
        label = labels[i]
        feature = {
            "val/label": _bytes_feature(tf.compat.as_bytes(label.tobytes())),  # label.tostring()以字节形式存在的字符串
            "val/audio": _bytes_feature(tf.compat.as_bytes(ado.tobytes())),
            "val/image": _bytes_feature(tf.compat.as_bytes(img.tobytes())),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()  # 刷新缓存区
    print(len(image_list), "validation data saved.. ")
    predict_val(NUM_VAL,PER_VAL)
val_acc()
