import glob
import pickle
import sys
from random import shuffle

import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import tensorflow as tf
from python_speech_features import logfbank


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


def process_wav(wav_file):
    (rate, sig) = wav.read(wav_file)  # 674816
    # print(len(sig))

    '''
     提取logfbank特征
     logfbank(signal, samplerate=16000, winlen=0.025, winstep=0.01, nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97)
     nfft – FFT大小。默认值为512。
     '''
    fbank_feat = logfbank(sig, rate, 0.025,0.01,26,1024)  #, 0.025,0.01,26,2048
    # fbank_feat.shape = (3059,26)
    # Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
    a = fbank_feat.flatten()
    # reshape()是数组array中的方法，作用是将数据重新组织 让a变成一维数组
    single_vec_feat = a.reshape(1, -1)  # single_vec_feat.shape = (1,79534)
    #print(single_vec_feat.shape)
    if single_vec_feat.shape != (1,79534):
        print(single_vec_feat.shape,wav_file)
    # single_vec_feat = single_vec_feat.astype(np.float64)
    return single_vec_feat



def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


df = load_pickle("Annotations/annotation_training.pkl")
NUM_VID = len(df)
addrs = []
labels = []
for i in range(NUM_VID):
    filelist = glob.glob(
        "VoiceData/trainingData/" + (df["VideoName"].iloc[i]).split(".mp4")[0] + ".wav"
    )
    addrs += filelist
    new = [np.array(df.drop(["VideoName"], 1, inplace=False).iloc[i]).astype(np.float32)]
    # print(i,filelist,new)
    labels += new

c = list(zip(addrs, labels))
shuffle(c)
train_addrs, train_labels = zip(*c)
# train_addrs, train_labels = addrs , labels
train_filename = "Logf_audio_train.tfrecords"  # address to save the TFRecords file
# open the TFRecords file
writer = tf.compat.v1.python_io.TFRecordWriter(train_filename)
other_audio = process_wav(train_addrs[0])
other_label = train_labels[0]
for i in range(len(train_addrs)):
    # print how many audio are saved every 1000 images
    if not i % 1000:
        print("Train data: {}/{}".format(i, len(train_addrs)))
        sys.stdout.flush()
    # Load the audio
    audio = process_wav(train_addrs[i])
    label = train_labels[i]
    if audio.shape != (1,79534):
        audio = other_audio
        label = other_label
        print('new_audio:', audio.shape)

    # Create a feature
    feature = {
        "train/audio": _bytes_feature(tf.compat.as_bytes(audio.tobytes())),
        "train/label": _bytes_feature(tf.compat.as_bytes(label.tobytes())),
    }
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    writer.write(example.SerializeToString())


writer.close()
sys.stdout.flush()

print(len(train_addrs), "training audio files saved.. ")

df = load_pickle("Annotations/annotation_validation.pkl")
NUM_VID = len(df)
addrs = []
labels = []
for i in range(NUM_VID):
    filelist = glob.glob(
        "VoiceData/validationData/"
        + (df["VideoName"].iloc[i]).split(".mp4")[0]
        + ".wav"
    )
    addrs += filelist
    labels += [np.array(df.drop(["VideoName"], 1, inplace=False).iloc[i]).astype(np.float32)]

c = list(zip(addrs, labels))
shuffle(c)
val_addrs, val_labels = zip(*c)
val_filename = "Logf_audio_val.tfrecords"  # address to save the TFRecords file
# module 'tensorflow' has no attribute 'python_io'
# 由于版本升级问题，版本2.0的函数文件路径发生变化。
# 解决办法：在python前面加上compat.v1.。
writer = tf.compat.v1.python_io.TFRecordWriter(val_filename)  # open the TFRecords file
other_val_audio = process_wav(val_addrs[0])
other_val_label = labels[0]
for i in range(len(val_addrs)):
    # print how many audio are saved every 1000 images
    if not i % 1000:
        print("val data: {}/{}".format(i, len(val_addrs)))
        sys.stdout.flush()
    # Load the audio
    audio = process_wav(val_addrs[i])
    label = val_labels[i]
    if audio.shape != (1,79534):
        audio = other_val_audio
        label = other_val_label
        print('new_val_audio:', audio.shape)
    # Create a feature
    feature = {
        "val/audio": _bytes_feature(tf.compat.as_bytes(audio.tobytes())),
        "val/label": _bytes_feature(tf.compat.as_bytes(label.tobytes())),
    }
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    writer.write(example.SerializeToString())


writer.close()
sys.stdout.flush()

print(len(val_addrs), "val audio files saved.. ")