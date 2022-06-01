import pickle
import warnings
import glob
import sys
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.io.wavfile as wav


from random import shuffle

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





## 随机采样
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
        print("wav Shape error!")
    return b

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def test_data_to_TF(per_video=48):
    df = load_pickle("Annotations/annotation_training.pkl")
    NUM_VID = len(df)  # 6000=75*80
    wav_addrs = []
    labels = []
    for i in range(NUM_VID):
        wav_name = glob.glob(
            "VoiceData/trainingData/" + (df["VideoName"].iloc[i]).split(".mp4")[0] + ".wav"
        )
        wav_addrs += [
                         wav_name[0]
                     ] * per_video  # 音频列表
        labels += [
                      np.array(df.drop(["VideoName"], 1, inplace=False).iloc[i]).astype(np.float32)
                  ] * per_video  # 标签列表

    if (len(wav_addrs) != per_video * NUM_VID):
        print("TRAIN DATA LENGTH ERROR: ", len(wav_addrs))
        return

    c = list(zip(wav_addrs, labels))  # zip函数作用：依次取一个元组，组成一个元组  即将图片和标签一一对应打包
    shuffle(c)  # 将序列中的元素随机排序防止一个视频的n个照片在一起
    train_wav_addrs, train_labels = zip(*c)  # 再把图片、音频和标注分别取出保存


    train_filename = "Res34_train.tfrecords"  # address to save the TFRecords file
    # open the TFRecords file
    writer = tf.io.TFRecordWriter(train_filename)
    # 每一次写入一条样本记录
    for i in range(len(train_wav_addrs)):  # 把图片和对应标注一次写入tfrecords文件
        # 每写入两千条 打印一次
        if not i % 2000:
            print("Train data of epoch: {}/{} ".format(i, len(train_wav_addrs)))
            sys.stdout.flush()
        ado = process_wav(train_wav_addrs[i])
        label = train_labels[i]
        # Create a feature 每一条样本的特征，将一系列特征组织成一条样本
        feature = {
            "train/label": _bytes_feature(tf.compat.as_bytes(label.tobytes())),  # label.tostring()以字节形式存在的字符串
            "train/audio": _bytes_feature(tf.compat.as_bytes(ado.tobytes())),
        }
        # Create an example protocol buffer 每一条样本的特征，将一系列特征组织成一条样本
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file  将每一条样本写入到tfrecord文件
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()  # 刷新缓存区
    print(len(train_labels), "training data saved.. ")
    ##### TRAINING DATA ####


def val_data_to_TF(per_video=32):
    df = load_pickle("Annotations/annotation_validation.pkl")
    NUM_VID = len(df)  # 2000
    wav_addrs = []
    labels = []
    for i in range(NUM_VID):  # NUM_VID
        # 获取指定目录下的图片
        wav_name = glob.glob(
            "VoiceData/validationData/" + (df["VideoName"].iloc[i]).split(".mp4")[0] + ".wav"
        )
        wav_addrs += [
                         wav_name[0]
                     ] * per_video  # 音频列表
        labels += [
                      np.array(df.drop(["VideoName"], 1, inplace=False).iloc[i]).astype(np.float32)
                  ] * per_video  # 标签列表

    if (len(wav_addrs) != per_video * NUM_VID):
        print("VAL DATA LENGTH ERROR:", len(wav_addrs))
        return

    c = list(zip(wav_addrs, labels))  # zip函数作用：依次取一个元组，组成一个元组  即将图片和标签一一对应打包
    shuffle(c)  # 将序列中的元素随机排序防止一个视频的n个照片在一起
    val_wav_addrs, val_labels = zip(*c)  # 再把图片、音频和标注分别取出保存

    val_filename = "Res34_val.tfrecords"  # address to save the TFRecords file
    # open the TFRecords file
    writer = tf.io.TFRecordWriter(val_filename)
    # 每一次写入一条样本记录
    for i in range(len(val_wav_addrs)):  # 把图片和对应标注一次写入tfrecords文件
        # 每写入两千条 打印一次
        if not i % 2000:
            print("Validation data of epoch: {}/{} ".format(i, len(val_wav_addrs)))
            sys.stdout.flush()

        ado = process_wav(val_wav_addrs[i])
        label = val_labels[i]
        # Create a feature 每一条样本的特征，将一系列特征组织成一条样本
        feature = {
            "val/label": _bytes_feature(tf.compat.as_bytes(label.tobytes())),  # label.tostring()以字节形式存在的字符串
            "val/audio": _bytes_feature(tf.compat.as_bytes(ado.tobytes())),
        }
        # Create an example protocol buffer 每一条样本的特征，将一系列特征组织成一条样本
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file  将每一条样本写入到tfrecord文件
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()  # 刷新缓存区
    print(len(val_labels), "validation data saved.. ")
    ##### VALIDATION DATA ####


test_data_to_TF()
val_data_to_TF()
