import glob
import pickle
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from random import shuffle
from PIL import Image


# 知道图片弄成想要的尺寸 和 编码方式   224*224
def load_image(addr):
    img = np.array(Image.open(addr).resize((224, 224), Image.ANTIALIAS))
    img = img.astype(np.uint8)
    return img




def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


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


##### TRAINING DATA ####
def tr_data_to_tf():
    df = load_pickle("Annotations/annotation_training.pkl")
    NUM_VID =len(df) #6000=75*80
    per_video = 24
    labels = []
    filelist = []
    for i in range(NUM_VID):
        # 获取指定目录下的图片
        index = 1
        for j in range(per_video):
            image_name = glob.glob(
                "ImageData/trainingData/"
                + (df["VideoName"].iloc[i]).split(".mp4")[0]
                + "/frame" + str(index) + ".jpg"
            )
            filelist += [image_name[0]]
            index += 4

        labels += [
                      np.array(df.drop(["VideoName"], 1, inplace=False).iloc[i]).astype(np.float32)
                  ] * per_video   # 标签列表

    if len(filelist) != per_video * NUM_VID or len(labels)!= per_video * NUM_VID:
        print("Train image data length error:", len(filelist), len(labels))
        return

    c = list(zip(filelist, labels))  # zip函数作用：依次取一个元组，组成一个元组  即将图片和标签一一对应打包
    shuffle(c)  # 将序列中的元素随机排序防止一个视频的n个照片在一起
    train_addrs, train_labels = zip(*c)  # 再把图片和标注分别取出保存  train_addrs=591937

    ##  tfrecords文件是以二进制进行存储的，适合以串行的方式读取大批量的数据。
    ##  对于训练数据而言，我们可以编写程序将普通的训练数据保存为tfrecords数据格式
    train_filename = "Dan_image_train.tfrecords"  # address to save the TFRecords file
    # open the TFRecords file
    writer = tf.io.TFRecordWriter(train_filename)
    # 每一次写入一条样本记录
    for i in range(len(train_addrs)):  # range(325000, len(train_addrs))   把图片和对应标注一次写入tfrecords文件
        # 每写入两千条 打印一次
        if not i % 2000:
            print("Train data: {}/{}".format(i, len(train_addrs)))
            sys.stdout.flush()
        # Load the image  加载图片
        img = load_image(train_addrs[i])
        label = train_labels[i]
        # Create a feature 每一条样本的特征，将一系列特征组织成一条样本
        feature = {
            "train/label": _bytes_feature(tf.compat.as_bytes(label.tobytes())),  # label.tostring()以字节形式存在的字符串
            "train/image": _bytes_feature(tf.compat.as_bytes(img.tobytes())),
        }
        # Create an example protocol buffer 每一条样本的特征，将一系列特征组织成一条样本
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file  将每一条样本写入到tfrecord文件
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()  # 刷新缓存区

    print(len(train_addrs), "training images saved.. ")
##### TRAINING DATA ####



##### VALIDATION DATA ####
def val_data_to_tf():
    df = load_pickle("Annotations/annotation_validation.pkl")
    NUM_VID =len(df)  #2000
    per_video = 24
    filelist = []
    labels = []
    for i in range(NUM_VID):
        # 获取指定目录下的图片
        index = 1
        for j in range(per_video):
            image_name = glob.glob(
                "ImageData/validationData/"
                + (df["VideoName"].iloc[i]).split(".mp4")[0]
                + "/frame" + str(index) + ".jpg"
            )
            filelist += [image_name[0]]
            index += 4

        labels += [
                      np.array(df.drop(["VideoName"], 1, inplace=False).iloc[i]).astype(np.float32)
                  ] * per_video  # 标签列表

    if len(filelist) != per_video * NUM_VID or len(labels) != per_video * NUM_VID:
        print("Val image data length error:", len(filelist), len(labels))
        return

    c = list(zip(filelist, labels))
    shuffle(c)
    val_addrs, val_labels = zip(*c)

    val_filename = "Dan_image_val.tfrecords"  # address to save the TFRecords file
    # open the TFRecords file
    writer = tf.io.TFRecordWriter(val_filename)

    for i in range(len(val_addrs)):
        # print how many images are saved every 2000 images
        if not i % 2000:
            print("Val data: {}/{}".format(i, len(val_addrs)))
            sys.stdout.flush()
        # Load the image
        img = load_image(val_addrs[i])
        label = val_labels[i].astype(np.float32)
        feature = {
            "val/label": _bytes_feature(tf.compat.as_bytes(label.tobytes())),
            "val/image": _bytes_feature(tf.compat.as_bytes(img.tobytes())),
        }
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()
    print(len(val_addrs), "validation images saved.. ")

##### VALIDATION DATA ####

tr_data_to_tf()
val_data_to_tf()
