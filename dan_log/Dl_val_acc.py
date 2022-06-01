import glob
import os
import sys
import warnings
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.io.wavfile as wav

from PIL import Image
from dan_plus import DAN_PLUS
from Logf_reg import Audio_reg
from python_speech_features import logfbank

tf.compat.v1.disable_eager_execution()

warnings.filterwarnings("ignore")

REG_PENALTY = 0


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
    (rate, sig) = wav.read(wav_file)
    fbank_feat = logfbank(sig, rate, 0.025, 0.01, 26, 1024)  # , 0.025,0.01,26,2048
    a = fbank_feat.flatten()
    single_vec_feat = a.reshape(1, -1)
    if single_vec_feat.shape != (1, 79534):
        print(single_vec_feat.shape, wav_file)
    return single_vec_feat

def load_image(addr):
    img = np.array(Image.open(addr).resize((224, 224), Image.ANTIALIAS))
    img = img.astype(np.uint8)
    return img

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def predict_image(NUM_VAL, PER_VAL):

    BATCH_SIZE = PER_VAL # 24
    NUM_IMAGES = NUM_VAL * PER_VAL
    file_list = ["param/param_dl_imgs1.pkl",
                 "param/param_dl_imgs2.pkl",
                 "param/param_dl_imgs3.pkl",
                 "param/param_dl_imgs4.pkl",
                 "param/param_dl_imgs5.pkl"]

    imgs = tf.compat.v1.placeholder("float", [None, 224, 224, 3], name="image_placeholder")
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.compat.v1.Session(config=config) as sess:

        model = DAN_PLUS(imgs, REG_PENALTY=REG_PENALTY, preprocess="vggface")

        val_filename_queue = tf.compat.v1.train.string_input_producer(
            ["val_acc/Dan_log/val_images.tfrecords"], num_epochs=len(file_list)
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
            print("-------------------epoch ", j + 1, "------------------")
            i = 0
            error = 0
            per_acc = []
            val_imgs_output = []
            val_acc_list = []
            val_mean_acc_list = []

            while i < NUM_IMAGES:
                i += BATCH_SIZE
                try:
                    epoch_x, epoch_y = sess.run([val_images, val_labels])
                except:
                    print(error, ": Error in reading this batch", int(i/BATCH_SIZE))
                    if error >= 10:
                        break
                    error += 1
                    continue

                output = sess.run(
                    [model.output], feed_dict={imgs: epoch_x.astype(np.float32)}
                )
                val_imgs_output.append(np.mean(output[0], axis=0))
                per_acc.append(np.mean(1 - np.absolute(output[0] - epoch_y), axis=0))
                mean_acc = np.mean(per_acc)
                val_acc_list.append(per_acc)
                val_mean_acc_list.append(mean_acc)

            if (len(val_mean_acc_list) != NUM_VAL or len(val_acc_list) != NUM_VAL):
                print("IMAGE ERROR")

            val_acc = np.mean(np.concatenate(val_acc_list), axis=0)
            val_mean_acc = np.mean(val_mean_acc_list)
            total_imgs_output += [val_imgs_output]
            print("val_image_acc: ", val_acc)
            print("val_image_mean_acc: ", val_mean_acc)
            print("----------------------------------------------")


        coord.request_stop()
        coord.join(threads)

    return total_imgs_output



def predict_audio(NUM_VAL):

    NUM_AUDIOS = NUM_VAL

    ados = tf.compat.v1.placeholder("float", [1, 79534], name="audio_placeholder")
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.compat.v1.Session(config=config) as sess:

        model = Audio_reg(ados, REG_PENALTY=REG_PENALTY)

        val_filename_queue = tf.compat.v1.train.string_input_producer(
            ["val_acc/Dan_log/val_audios.tfrecords"], num_epochs=1
        )
        val_reader = tf.compat.v1.TFRecordReader()
        _, val_serialized_example = val_reader.read(val_filename_queue)


        val_feature = {
            "val/audio": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
            "val/label": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
        }
        val_features = tf.compat.v1.parse_single_example(val_serialized_example, features=val_feature)

        val_audio = tf.compat.v1.decode_raw(val_features["val/audio"], tf.compat.v1.float64)
        val_label = tf.compat.v1.decode_raw(val_features["val/label"], tf.compat.v1.float32)

        val_audio = tf.compat.v1.reshape(val_audio, [1, 79534])
        val_label = tf.compat.v1.reshape(val_label, [5])

        init_op = tf.compat.v1.group(
            tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()
        )
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord)

        model.load_trained_model("param/param_dl_ados.pkl", sess)

        val_ados_output = []
        val_acc_list = []
        val_mean_acc_list = []

        i = 0
        error = 0
        while i < NUM_AUDIOS:
            i += 1
            try:
                epoch_x, epoch_y = sess.run([val_audio, val_label])
            except:
                print(error, ": Error in reading this batch", i)
                if error >= 10:
                    break
                error += 1
                continue

            output = sess.run(
                [model.output], feed_dict={ados: epoch_x.astype(np.float64)}
            )
            output = output[0].reshape(5)
            val_ados_output.append(output)
            acc = 1 - np.absolute(output - epoch_y)
            mean_acc = np.mean(acc)
            val_acc_list.append(acc)
            val_mean_acc_list.append(mean_acc)


        coord.request_stop()
        coord.join(threads)

    if (len(val_mean_acc_list) != NUM_VAL or len(val_acc_list) != NUM_VAL):
        print("AUDIO ERROR")


    val_acc = np.mean(val_acc_list, axis=0)
    val_mean_acc = np.mean(val_mean_acc_list)
    print("val_audio_acc: ", val_acc)
    print("val_audio_mean_acc: ", val_mean_acc)

    return val_ados_output


def val_mean_acc():
    df = load_pickle("Annotations/annotation_validation.pkl")
    NUM_VID = len(df)  # 2000
    NUM_VAL = NUM_VID
    PER_VAL = 24
    image_list = []
    audio_list = []
    labels_img = []
    labels_ado = []
    for i in range(NUM_VAL):
        index = 1
        for j in range(PER_VAL):

            image_name = glob.glob(
                "ImageData/validationData/"
                + (df["VideoName"].iloc[i]).split(".mp4")[0]
                + "/frame" + str(index) + ".jpg"
            )
            index += 4
            image_list += [image_name[0]]


        labels_img += [np.array(df.drop(["VideoName"], 1, inplace=False).iloc[i]).astype(np.float32)] * PER_VAL

        audio_name = glob.glob(
            "VoiceData/validationData/"
            + (df["VideoName"].iloc[i]).split(".mp4")[0]
            + ".wav"
        )
        audio_list += [audio_name[0]]
        labels_ado += [np.array(df.drop(["VideoName"], 1, inplace=False).iloc[i]).astype(np.float32)]

    if len(image_list) != PER_VAL * NUM_VAL or len(audio_list) != NUM_VAL:
        print("data length error!", len(image_list), len(audio_list))
        return

    try:
        os.makedirs("val_acc/Dan_log")
    except OSError:
        print("Error: Creating val_acc/Dan_log")

    val_img_filename = "val_acc/Dan_log/val_images.tfrecords"  # address to save the TFRecords file
    writer_img = tf.io.TFRecordWriter(val_img_filename)
    for i in range(len(image_list)):
        img = load_image(image_list[i])
        label = labels_img[i]
        feature = {
            "val/label": _bytes_feature(tf.compat.as_bytes(label.tobytes())),  # label.tostring()以字节形式存在的字符串
            "val/image": _bytes_feature(tf.compat.as_bytes(img.tobytes())),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer_img.write(example.SerializeToString())
    writer_img.close()
    sys.stdout.flush()  # 刷新缓存区
    print(len(image_list), "validation images saved.. ")

    val_ado_filename = "val_acc/Dan_log/val_audios.tfrecords"
    writer_ado = tf.io.TFRecordWriter(val_ado_filename)
    ado_other = process_wav(audio_list[0])
    for i in range(len(audio_list)):
        ado = process_wav(audio_list[i])
        if ado.shape!= (1, 79534):
            # print("New ado!")
            ado = ado_other

        label = labels_ado[i]
        feature = {
            "val/label": _bytes_feature(tf.compat.as_bytes(label.tobytes())),  # label.tostring()以字节形式存在的字符串
            "val/audio": _bytes_feature(tf.compat.as_bytes(ado.tobytes())),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer_ado.write(example.SerializeToString())

    writer_ado.close()
    sys.stdout.flush()  # 刷新缓存区
    print(len(audio_list), "validation audios saved.. ")


    imgs_output = predict_image(NUM_VAL, PER_VAL)
    ados_output = predict_audio(NUM_VAL)
    print("\n\nThe combine accuracy of images and audios：")
    for j in range(len(imgs_output)):
        output = np.add(imgs_output[j], ados_output)
        new_output = np.array(output) / 2
        acc = np.mean(1 - np.absolute(new_output - labels_ado), axis=0)
        mean_acc = np.mean(acc)
        print("-------------------epoch ", j + 1, "------------------")
        print("val_acc: ", acc)
        print("val_mean_acc: ", mean_acc)
        print("----------------------------------------------")


val_mean_acc()