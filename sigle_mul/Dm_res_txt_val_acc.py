import glob
import os
import random
import sys
import warnings
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.io.wavfile as wav

from PIL import Image
from Dan_max import image_net
from Resnet_34 import audio_net
from text_reg import Text_reg
from sklearn.feature_extraction.text import TfidfVectorizer

tf.compat.v1.disable_eager_execution()

warnings.filterwarnings("ignore")

REG_PENALTY = 0



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
    tr_vec.fit_transform(tr_word_list)
    val_vec = TfidfVectorizer(vocabulary=tr_vec.vocabulary_)
    val_word_count = val_vec.fit_transform(val_word_list)
    val_word_vec = val_word_count.toarray()
    val_text = dict(zip(val_video_name, val_word_vec))
    return val_text


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

def load_image(addr):
    img = np.array(Image.open(addr).resize((224, 224), Image.ANTIALIAS))
    img = img.astype(np.uint8)
    return img

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def predict_image(NUM_VAL, PER_VAL):

    BATCH_SIZE = PER_VAL # 32
    NUM_IMAGES = NUM_VAL * PER_VAL
    file_list = ["param/param_dan_max1.pkl",
                 "param/param_dan_max2.pkl",
                 "param/param_dan_max3.pkl",
                 "param/param_dan_max4.pkl"]

    imgs = tf.compat.v1.placeholder("float", [None, 224, 224, 3], name="image_placeholder")
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.compat.v1.Session(config=config) as sess:

        model = image_net(imgs, REG_PENALTY=REG_PENALTY)

        val_filename_queue = tf.compat.v1.train.string_input_producer(
            ["val_acc/Dan_max_res/val_images.tfrecords"], num_epochs=len(file_list)
        )
        val_reader = tf.compat.v1.TFRecordReader()
        _, val_serialized_example = val_reader.read(val_filename_queue)

        val_feature = {
            "val/image": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
            "val/label": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
        }
        val_features = tf.compat.v1.parse_single_example(val_serialized_example, features=val_feature)

        val_image = tf.compat.v1.decode_raw(val_features["val/image"], tf.compat.v1.uint8)
        val_label = tf.compat.v1.decode_raw(val_features["val/label"], tf.compat.v1.float32)

        val_image = tf.compat.v1.reshape(val_image, [224, 224, 3])
        val_label = tf.compat.v1.reshape(val_label, [5])

        # 不打乱
        val_images, val_labels = tf.compat.v1.train.batch(
            [val_image, val_label],
            batch_size=BATCH_SIZE,
            capacity=200,
            allow_smaller_final_batch=True,
        )

        init_op = tf.compat.v1.group(
            tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()
        )
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord)

        print("Length of total image val:", NUM_IMAGES)
        print("Num of epoch:", len(file_list))
        total_imgs_output = []
        for j in range(len(file_list)):
            # 加载训练好的参数
            model.load_trained_model(file_list[j], sess)
            print("---------------image epoch ", j + 1, "-----------------")
            i = 0
            error = 0
            val_imgs_output = []
            val_acc_list = []
            val_mean_acc_list = []

            while i < NUM_IMAGES:
                i += BATCH_SIZE
                try:
                    epoch_x, epoch_y = sess.run([val_images, val_labels])
                except:
                    print(error+1, ": Error in reading batch_", int(i/BATCH_SIZE))
                    if error >= 10:
                        break
                    error += 1
                    continue

                output = sess.run(
                    [model.output], feed_dict={imgs: epoch_x.astype(np.float32)}
                )
                val_imgs_output.append(np.mean(output[0], axis=0))
                acc = np.mean(1 - np.absolute(output[0] - epoch_y), axis=0)
                mean_acc = np.mean(acc)
                val_acc_list.append(acc)
                val_mean_acc_list.append(mean_acc)

            if (len(val_mean_acc_list) != NUM_VAL or len(val_acc_list) != NUM_VAL):
                print("IMAGE ERROR")

            val_acc = np.mean(val_acc_list, axis=0)
            val_mean_acc = np.mean(val_mean_acc_list)
            total_imgs_output += [val_imgs_output]
            print("val_image_acc: ", val_acc)
            print("val_image_mean_acc: ", val_mean_acc)
            print("-----------------------------------------------")


        coord.request_stop()
        coord.join(threads)

    if len(total_imgs_output) != len(file_list):
        print("Image epoch error!")

    return total_imgs_output



def predict_audio(NUM_VAL,PER_VAL):

    BATCH_SIZE = PER_VAL  # 32
    NUM_AUDIOS = NUM_VAL * PER_VAL

    audio_file_list = ["param/param_res34_ados1.pkl",
                       "param/param_res34_ados2.pkl",
                       "param/param_res34_ados3.pkl",
                       "param/param_res34_ados4.pkl"]

    ados = tf.compat.v1.placeholder("float", [None, 1, 50176, 1], name="audio_placeholder")
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.compat.v1.Session(config=config) as sess:

        model = audio_net(ados, REG_PENALTY=REG_PENALTY,is_training=True)

        val_filename_queue = tf.compat.v1.train.string_input_producer(
            ["val_acc/Dan_max_res/val_audios.tfrecords"], num_epochs=len(audio_file_list)
        )
        val_reader = tf.compat.v1.TFRecordReader()
        _, val_serialized_example = val_reader.read(val_filename_queue)


        val_feature = {
            "val/audio": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
            "val/label": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
        }
        val_features = tf.compat.v1.parse_single_example(val_serialized_example, features=val_feature)

        val_audio = tf.compat.v1.decode_raw(val_features["val/audio"], tf.compat.v1.float32)
        val_label = tf.compat.v1.decode_raw(val_features["val/label"], tf.compat.v1.float32)

        val_audio = tf.compat.v1.reshape(val_audio, [1, 50176, 1])
        val_label = tf.compat.v1.reshape(val_label, [5])

        val_audios, val_labels = tf.compat.v1.train.batch(
            [val_audio, val_label],
            batch_size=BATCH_SIZE,
            capacity=200,
            allow_smaller_final_batch=True,
        )

        init_op = tf.compat.v1.group(
            tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()
        )
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord)

        print("\nLength of total audio val:", NUM_AUDIOS)
        print("Num of epoch:", len(audio_file_list))
        total_ados_output = []
        for j in range(len(audio_file_list)):
            model.load_trained_model(audio_file_list[j], sess)

            print("---------------audio epoch ", j + 1, "-----------------")
            val_ados_output = []
            val_acc_list = []
            val_mean_acc_list = []
            i = 0
            error = 0
            while i < NUM_AUDIOS:
                i += BATCH_SIZE
                try:
                    epoch_x, epoch_y = sess.run([val_audios, val_labels])
                except:
                    print(error+1, ": Error in reading  batch_", int(i/BATCH_SIZE))
                    if error >= 10:
                        break
                    error += 1
                    continue

                output = sess.run(
                    [model.output], feed_dict={ados: epoch_x.astype(np.float32)}
                )

                per_output = np.mean(output[0],axis=0)
                val_ados_output.append(per_output)
                acc = np.mean(1 - np.absolute(output[0] - epoch_y),axis=0)
                mean_acc = np.mean(acc)
                val_acc_list.append(acc)
                val_mean_acc_list.append(mean_acc)


            if (len(val_mean_acc_list) != NUM_VAL or len(val_acc_list) != NUM_VAL):
                print("AUDIO ERROR")

            val_acc = np.mean(val_acc_list, axis=0)
            val_mean_acc = np.mean(val_mean_acc_list)
            total_ados_output.append(val_ados_output)
            print("val_audio_acc: ", val_acc)
            print("val_audio_mean_acc: ", val_mean_acc)
            print("-----------------------------------------------")

        coord.request_stop()
        coord.join(threads)


    if len(total_ados_output) != len(audio_file_list):
        print("Audio epoch error!")

    return total_ados_output


def predict_text(NUM_VAL):

    NUM_TEXTS = NUM_VAL

    texts = tf.compat.v1.placeholder("float", [1, 1, 11264, 1], name="text_placeholder")
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.compat.v1.Session(config=config) as sess:

        model = Text_reg(texts, REG_PENALTY=REG_PENALTY,is_training=False)

        val_filename_queue = tf.compat.v1.train.string_input_producer(
            ["val_acc/Dan_max_res/val_texts.tfrecords"], num_epochs=1
        )
        val_reader = tf.compat.v1.TFRecordReader()
        _, val_serialized_example = val_reader.read(val_filename_queue)


        val_feature = {
            "val/text": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
            "val/label": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
        }
        val_features = tf.compat.v1.parse_single_example(val_serialized_example, features=val_feature)

        val_text = tf.compat.v1.decode_raw(val_features["val/text"], tf.compat.v1.float64)
        val_label = tf.compat.v1.decode_raw(val_features["val/label"], tf.compat.v1.float32)

        val_text = tf.compat.v1.reshape(val_text, [1, 11264, 1])
        val_label = tf.compat.v1.reshape(val_label, [5])

        val_texts, val_labels = tf.compat.v1.train.batch(
            [val_text, val_label],
            batch_size=1,
            capacity=200,
            allow_smaller_final_batch=True,
        )

        init_op = tf.compat.v1.group(
            tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()
        )
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord)

        model.load_trained_model("param/param_text.pkl", sess)

        val_texts_output = []
        val_acc_list = []
        val_mean_acc_list = []

        i = 0
        error = 0
        while i < NUM_TEXTS:
            i += 1
            try:
                epoch_x, epoch_y = sess.run([val_texts, val_labels])
            except:
                print(error, ": Error in reading this batch", i)
                if error >= 10:
                    break
                error += 1
                continue

            output = sess.run(
                [model.output], feed_dict={texts: epoch_x.astype(np.float32)}
            )
            output = output[0].reshape(5)
            val_texts_output.append(output)
            acc = 1 - np.absolute(output - epoch_y)
            mean_acc = np.mean(acc)
            val_acc_list.append(acc)
            val_mean_acc_list.append(mean_acc)


        coord.request_stop()
        coord.join(threads)

    if (len(val_mean_acc_list) != NUM_VAL or len(val_acc_list) != NUM_VAL):
        print("TEXT ERROR")

    val_acc = np.mean(val_acc_list, axis=0)
    val_mean_acc = np.mean(val_mean_acc_list)
    print("val_text_acc: ", val_acc)
    print("val_text_mean_acc: ", val_mean_acc)

    return val_texts_output


def val_mean_acc():
    val_text = word_to_vector()
    df = load_pickle("Annotations/annotation_validation.pkl")
    NUM_VID = len(df)  # 2000
    NUM_VAL = NUM_VID
    PER_VAL = 32
    image_list = []
    audio_list = []
    labels = []
    text_labels = []
    text_list = []
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
        audio_list += [audio_name[0]]*PER_VAL
        labels += [np.array(df.drop(["VideoName"], 1, inplace=False).iloc[i]).astype(np.float32)] * PER_VAL
        text_labels += [np.array(df.drop(["VideoName"], 1, inplace=False).iloc[i]).astype(np.float32)]
        video_name = df["VideoName"].iloc[i]
        text_list += [val_text[video_name]]

    if len(image_list) != PER_VAL * NUM_VAL or len(audio_list) != PER_VAL * NUM_VAL or len(text_list) != NUM_VAL:
        print("data length error!", len(image_list), len(audio_list), len(text_list))
        return

    try:
        os.makedirs("val_acc/Dan_max_res")
    except OSError:
        print("Error: Creating val_acc/Dan_max_res")

    val_img_filename = "val_acc/Dan_max_res/val_images.tfrecords"  # address to save the TFRecords file
    writer_img = tf.io.TFRecordWriter(val_img_filename)
    for i in range(len(image_list)):
        img = load_image(image_list[i])
        label = labels[i]
        feature = {
            "val/label": _bytes_feature(tf.compat.as_bytes(label.tobytes())),  # label.tostring()以字节形式存在的字符串
            "val/image": _bytes_feature(tf.compat.as_bytes(img.tobytes())),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer_img.write(example.SerializeToString())
    writer_img.close()
    sys.stdout.flush()  # 刷新缓存区
    print(len(image_list), "validation images saved.. ")

    val_ado_filename = "val_acc/Dan_max_res/val_audios.tfrecords"
    writer_ado = tf.io.TFRecordWriter(val_ado_filename)
    for i in range(len(audio_list)):
        ado = process_wav(audio_list[i])

        label = labels[i]
        feature = {
            "val/label": _bytes_feature(tf.compat.as_bytes(label.tobytes())),  # label.tostring()以字节形式存在的字符串
            "val/audio": _bytes_feature(tf.compat.as_bytes(ado.tobytes())),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer_ado.write(example.SerializeToString())

    writer_ado.close()
    sys.stdout.flush()  # 刷新缓存区
    print(len(audio_list), "validation audios saved.. ")


    val_txt_filename = "val_acc/Dan_max_res/val_texts.tfrecords"
    writer_txt = tf.io.TFRecordWriter(val_txt_filename)
    for i in range(len(text_list)):
        text = text_list[i]
        label = text_labels[i]
        feature = {
            "val/label": _bytes_feature(tf.compat.as_bytes(label.tobytes())),  # label.tostring()以字节形式存在的字符串
            "val/text": _bytes_feature(tf.compat.as_bytes(text.tobytes())),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer_txt.write(example.SerializeToString())

    writer_txt.close()
    sys.stdout.flush()  # 刷新缓存区
    print(len(text_list), "validation texts saved.. ")

    imgs_output = predict_image(NUM_VAL, PER_VAL)
    ados_output = predict_audio(NUM_VAL,PER_VAL)
    txts_output = predict_text(NUM_VAL)



    print("\n\nThe combine accuracy of images , audios and texts：")
    for j in range(len(imgs_output)):
        output_1 = np.add(imgs_output[j], ados_output[j])
        output = np.add(output_1, txts_output)
        new_output = np.array(output) / 3
        acc = np.mean(1 - np.absolute(new_output - text_labels), axis=0)
        mean_acc = np.mean(acc)
        print("-------------------epoch ", j + 1, "------------------")
        print("val_acc: ", acc)
        print("val_mean_acc: ", mean_acc)
        print("----------------------------------------------")


    print("\nThe combine accuracy of images and audios：")
    for j in range(len(imgs_output)):
        output = np.add(imgs_output[j], ados_output[j])
        new_output = np.array(output) / 2
        acc = np.mean(1 - np.absolute(new_output - text_labels), axis=0)
        mean_acc = np.mean(acc)
        print("-------------------epoch ", j + 1, "------------------")
        print("val_acc: ", acc)
        print("val_mean_acc: ", mean_acc)
        print("----------------------------------------------")



    print("\nThe combine accuracy of images and text：")
    for j in range(len(imgs_output)):
        output = np.add(imgs_output[j], txts_output)
        new_output = np.array(output) / 2
        acc = np.mean(1 - np.absolute(new_output - text_labels), axis=0)
        mean_acc = np.mean(acc)
        print("-------------------epoch ", j + 1, "------------------")
        print("val_acc: ", acc)
        print("val_mean_acc: ", mean_acc)
        print("----------------------------------------------")

    print("\nThe combine accuracy of texts and audios：")
    for j in range(len(ados_output)):
        output = np.add(txts_output, ados_output[j])
        new_output = np.array(output) / 2
        acc = np.mean(1 - np.absolute(new_output - text_labels), axis=0)
        mean_acc = np.mean(acc)
        print("-------------------epoch ", j + 1, "------------------")
        print("val_acc: ", acc)
        print("val_mean_acc: ", mean_acc)
        print("----------------------------------------------")


val_mean_acc()