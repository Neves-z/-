import pickle
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer

def transcription_video(pickle_file):
    with open(pickle_file, "rb") as f:
        word_data = pickle.load(f, encoding="latin1")
        video_name = list(word_data.keys())
        value_list = list(word_data.values())

    return value_list, video_name

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

def word_to_vector():
    tr_word_list, tr_video_name = transcription_video("transcription/transcription_training.pkl")
    val_word_list, val_video_name = transcription_video("transcription/transcription_validation.pkl")

    tr_vec = TfidfVectorizer(min_df=1,max_features=11264)  #11264
    tr_word_count = tr_vec.fit_transform(tr_word_list)
    tr_word_vec = tr_word_count.toarray()
    tr_text = dict(zip(tr_video_name, tr_word_vec))
    val_vec = TfidfVectorizer(vocabulary=tr_vec.vocabulary_)
    val_word_count = val_vec.fit_transform(val_word_list)
    val_word_vec = val_word_count.toarray()
    val_text = dict(zip(val_video_name, val_word_vec))
    return tr_text, val_text


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def train_data_to_tf(tr_text):
    df = load_pickle("Annotations/annotation_training.pkl")
    NUM_VID = len(df)  # 6000=75*80
    texts = []
    labels = []
    for i in range(NUM_VID):
        labels += [np.array(df.drop(["VideoName"], 1, inplace=False).iloc[i]).astype(np.float32)]
        video_name = df["VideoName"].iloc[i]
        texts += [tr_text[video_name]]


    if len(labels) != NUM_VID or len(texts) != NUM_VID:
        print("TRAIN DATA LENGTH ERROR: ", len(labels), len(texts))
        return


    train_filename = "text_train.tfrecords"  # address to save the TFRecords file
    # open the TFRecords file
    writer = tf.io.TFRecordWriter(train_filename)
    # 每一次写入一条样本记录
    for i in range(len(texts)):  # 把图片和对应标注一次写入tfrecords文件
        # 每写入两千条 打印一次
        if not i % 2000:
            print("Train data of epoch: {}/{} ".format(i, len(labels)))
            sys.stdout.flush()
        # Load the image  加载图片
        text= texts[i]
        label = labels[i]
        # Create a feature 每一条样本的特征，将一系列特征组织成一条样本
        feature = {
            "train/label": _bytes_feature(tf.compat.as_bytes(label.tobytes())),  # label.tostring()以字节形式存在的字符串
            "train/text": _bytes_feature(tf.compat.as_bytes(text.tobytes())),
        }
        # Create an example protocol buffer 每一条样本的特征，将一系列特征组织成一条样本
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file  将每一条样本写入到tfrecord文件
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()  # 刷新缓存区
    print(len(labels), "training data saved.. ")
    ##### TRAINING DATA ####


def val_data_to_tf(val_text):
    df = load_pickle("Annotations/annotation_validation.pkl")
    NUM_VID = len(df)  # 2000
    texts = []
    labels = []
    for i in range(NUM_VID):
        labels += [np.array(df.drop(["VideoName"], 1, inplace=False).iloc[i]).astype(np.float32)]
        video_name = df["VideoName"].iloc[i]
        texts += [val_text[video_name]]


    if len(labels) != NUM_VID or len(texts) != NUM_VID:
        print("TRAIN DATA LENGTH ERROR: ", len(labels), len(texts))
        return


    train_filename = "text_val.tfrecords"  # address to save the TFRecords file
    # open the TFRecords file
    writer = tf.io.TFRecordWriter(train_filename)
    # 每一次写入一条样本记录
    for i in range(len(texts)):  # 把图片和对应标注一次写入tfrecords文件
        # 每写入两千条 打印一次
        if not i % 1000:
            print("Val data of epoch: {}/{} ".format(i, len(labels)))
            sys.stdout.flush()
        # Load the image  加载图片
        text= texts[i]
        label = labels[i]
        # Create a feature 每一条样本的特征，将一系列特征组织成一条样本
        feature = {
            "val/label": _bytes_feature(tf.compat.as_bytes(label.tobytes())),  # label.tostring()以字节形式存在的字符串
            "val/text": _bytes_feature(tf.compat.as_bytes(text.tobytes())),
        }
        # Create an example protocol buffer 每一条样本的特征，将一系列特征组织成一条样本
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file  将每一条样本写入到tfrecord文件
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()  # 刷新缓存区
    print(len(labels), "validation data saved.. ")
    ##### TRAINING DATA ####


tr_text, val_text = word_to_vector()
train_data_to_tf(tr_text)
val_data_to_tf(val_text)

