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
from dan import DAN
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


def predict_image(file_name):



    def load_image(addr):
        img = np.array(Image.open(addr).resize((224, 224), Image.ANTIALIAS))
        img = img.astype(np.uint8)
        return img


    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    filelist = glob.glob(str(file_name) + "/*.jpg")
    val_addrs = filelist
    try:
        os.makedirs("val_acc/val_image/")
    except OSError:
        print("Error: Creating directory of data")

    val_filename = "val_acc/val_image/val_image.tfrecords"  # address to save the TFRecords file
    writer = tf.compat.v1.python_io.TFRecordWriter(val_filename)
    for i in range(len(val_addrs)):
        # Load the image
        img = load_image(val_addrs[i])
        feature = {"val/image": _bytes_feature(tf.compat.as_bytes(img.tobytes()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

    BATCH_SIZE = 20
    NUM_IMAGES = 100

    imgs = tf.compat.v1.placeholder("float", [None, 224, 224, 3], name="image_placeholder")
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.compat.v1.Session(config=config) as sess:

        model = DAN(imgs, REG_PENALTY=REG_PENALTY, preprocess="vggface")
        tr_reader = tf.compat.v1.TFRecordReader()
        tr_filename_queue = tf.compat.v1.train.string_input_producer(
            ["val_acc/val_image/val_image.tfrecords"], num_epochs=1
        )
        _, tr_serialized_example = tr_reader.read(tr_filename_queue)
        tr_feature = {"val/image": tf.compat.v1.FixedLenFeature([], tf.string)}
        tr_features = tf.compat.v1.parse_single_example(
            tr_serialized_example, features=tr_feature
        )

        tr_image = tf.compat.v1.decode_raw(tr_features["val/image"], tf.uint8)
        tr_image = tf.reshape(tr_image, [224, 224, 3])
        tr_images = tf.compat.v1.train.shuffle_batch(
            [tr_image],
            batch_size=BATCH_SIZE,
            capacity=100,
            min_after_dequeue=BATCH_SIZE,
            allow_smaller_final_batch=True,
        )
        init_op = tf.group(
            tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()
        )
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord)
        file_list = ["param_image.pkl"]
        num = []
        for pickle_file in file_list:
            error = 0
            model.load_trained_model(pickle_file, sess)
            i = 0
            while i < NUM_IMAGES:
                i += BATCH_SIZE
                try:
                    epoch_x = sess.run(tr_images)
                except:
                    if error >= 5:
                        break
                    error += 1
                    continue
                output = sess.run(
                    [model.output], feed_dict={imgs: epoch_x.astype(np.float32)}
                )
                num.append(output[0])
        coord.request_stop()
        # Wait for threads to stop
        coord.join(threads)
    #  np.concatenate 拼接数组  np.mean（axis=0）计算每一列的均值 round（n）保留几位小数
    #print(np.concatenate(num))
    a = np.mean(np.concatenate(num), axis=0)
    return a


def predict_audio(file_name):
    def process_wav(wav_file):
        (rate, sig) = wav.read(wav_file)
        fbank_feat = logfbank(sig, rate, 0.025, 0.01, 26, 1024)  # , 0.025,0.01,26,2048
        a = fbank_feat.flatten()
        single_vec_feat = a.reshape(1, -1)
        if single_vec_feat.shape != (1, 79534):
            print(single_vec_feat.shape, wav_file)
        return single_vec_feat

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    try:
        os.makedirs("val_acc/val_audio/")
    except OSError:
        print("Error: Creating directory of data")
    try:
        os.remove("val_acc/val_audio/val_audio.tfrecords")
    except OSError:
        print("Error: The file is not exsist")
    # address to save the TFRecords file
    val_filename = "val_acc/val_audio/val_audio.tfrecords"
    # open the TFRecords file
    writer = tf.compat.v1.python_io.TFRecordWriter(val_filename)
    audio = process_wav(file_name)
    if audio.shape != (1, 79534):
        print("The shape of test audio is not (1, 79534)! ")
        return [0]

    # Create a feature
    feature = {
        "val/audio": _bytes_feature(tf.compat.as_bytes(audio.tobytes())),
    }
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
    writer.close()
    sys.stdout.flush()



    Ados = tf.compat.v1.placeholder("float", [1,79534], name="audio_placeholder")
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.compat.v1.Session(config=config) as sess:

        model = Audio_reg(Ados, REG_PENALTY=REG_PENALTY)
        tr_reader = tf.compat.v1.TFRecordReader()
        tr_filename_queue = tf.compat.v1.train.string_input_producer(
            ["val_acc/val_audio/val_audio.tfrecords"], num_epochs=1
        )
        _, tr_serialized_example = tr_reader.read(tr_filename_queue)
        tr_feature = {"val/audio": tf.compat.v1.FixedLenFeature([], tf.string)}
        tr_features = tf.compat.v1.parse_single_example(tr_serialized_example, features=tr_feature)
        tr_audio = tf.compat.v1.decode_raw(tr_features["val/audio"], tf.compat.v1.float64)
        # Reshape image data into the original shape
        tr_audio = tf.compat.v1.reshape(tr_audio, [1, 79534])
        init_op = tf.group(
            tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()
        )
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord)



        file_list = ["param_audio.pkl"]
        for pickle_file in file_list:
            model.load_trained_model(pickle_file, sess)
            try:
                epoch_x = sess.run(tr_audio)
            except:
                print("Error")
                return
            output = sess.run(
                [model.output], feed_dict={Ados: epoch_x.astype(np.float64)}
            )
        coord.request_stop()
        # Wait for threads to stop
        coord.join(threads)
    a = output[0]
    return a[0]



def val_mean_acc():
    df = load_pickle("Annotations/annotation_validation.pkl")
    NUM_VID = len(df)  # 6000=75*80
    val_mean_acc = []
    val_acc = []
    output_image = []
    output_audio = []
    for i in range(NUM_VID):
        # 获取指定目录下的图片
        # print(df["VideoName"].iloc[i])
        val_image_filelist = glob.glob(
            "ImageData/validationData/"
            + df["VideoName"].iloc[i].split(".mp4")[0]
        )
        val_audio_filelist = glob.glob(
            "VoiceData/validationData/"
            + (df["VideoName"].iloc[i]).split(".mp4")[0]
            + ".wav"
        )
        label = np.array(df.drop(["VideoName"], 1, inplace=False).iloc[i]).astype(np.float32)
        val_image_filename = val_image_filelist[0]
        val_audio_filename = val_audio_filelist[0]

        out_img = predict_image(val_image_filename)
        output_image.append(out_img)
        out_audio = predict_audio(val_audio_filename)

        output = []
        output.append(out_img)
        if len(out_audio) == 5:
            output_audio.append(out_audio)
            output.append(out_audio)
        else:
            print("The shape of audio is not (1, 79534).")

        acc =1 - np.absolute(label - np.round(np.mean(output, axis=0), 3))
        mean_acc = np.mean(acc)
        print("acc: ", acc)
        print("mean_acc: ", mean_acc)
        val_acc += [acc]
        val_mean_acc += [mean_acc]


    val_acc = np.mean(val_acc, axis=0)
    val_mean_acc =np.mean(val_mean_acc, axis=0)
    print("val_acc: ", val_acc)
    print("val_mean_acc: ", val_mean_acc)

val_mean_acc()