import glob
import os
import sys
import cv2
import random
import warnings
import subprocess

import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav


from PIL import Image
from Resnet import Resnet_18

tf.compat.v1.disable_eager_execution()

warnings.filterwarnings("ignore")
BATCH_SIZE = 25


def load_image(addr):
    img = np.array(Image.open(addr).resize((224, 224), Image.ANTIALIAS))
    img = img.astype(np.uint8)
    return img

"""
# 固定采样
def process_wav(wav_file):
    (rate, sig) = wav.read(wav_file)  # 674816
    # print(len(sig))
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


    # print(j) # 652288
    a = np.concatenate(val_wav)
    b = a.reshape(1, 50176, 1)
    b = b.astype(np.float32)
    # print(b.shape)
    if b.shape != (1, 50176, 1):
        print("wav error!")
    return b
"""

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
        print("wav length error!")
    return b


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def data_process(file_name):

    cap = cv2.VideoCapture(file_name)

    file_name = (file_name.split(".mp4"))[0]

    try:
        os.makedirs("ImageData/testingData/" + file_name)
    except OSError:
        print("Error: Creating directory of data")


    cap.set(cv2.CAP_PROP_FRAME_COUNT, 33)
    length = 33
    count = 0
    ## Running a loop to each frame and saving it in the created folder
    while cap.isOpened():
        count += 1
        if length == count:
            break
        _, frame = cap.read()
        if frame is None:
            continue

        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_CUBIC)

        name = (
                "ImageData/testingData/" + str(file_name) + "/frame" + str(count) + ".jpg"
        )
        cv2.imwrite(name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    try:
        if not os.path.exists("VoiceData/testingData/"):
            os.makedirs("VoiceData/testingData/")
    except OSError:
        print("Error: Creating directory of data")
    command = "ffmpeg -i {}.mp4 -ab 320k -ac 2 -ar 44100 -vn VoiceData/testingData/{}.wav".format(
        file_name, file_name
    )
    subprocess.call(command, shell=True)

    image_addrs = []
    filelist = glob.glob("ImageData/testingData/" + str(file_name) + "/*.jpg")
    image_addrs += filelist

    audio_addr = glob.glob("VoiceData/testingData/" + str(file_name) + ".wav")

    try:
        os.makedirs("predict/res")
    except OSError:
        print("Error: Creating predict/res")

    predict_filename = "predict/res/predict_res.tfrecords"  # address to save the TFRecords file

    writer = tf.compat.v1.python_io.TFRecordWriter(predict_filename)
    for i in range(len(image_addrs)):
        # Load the image
        img = load_image(image_addrs[i])
        ado = process_wav(audio_addr[0])
        feature = {
            "predict/audio": _bytes_feature(tf.compat.as_bytes(ado.tobytes())),
            "predict/image": _bytes_feature(tf.compat.as_bytes(img.tobytes())),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()
    print(file_name, len(image_addrs),  "predicting data saved.. ")
    return len(image_addrs)

def predict(file_name):

    images_num = data_process(file_name)

    imgs = tf.compat.v1.placeholder("float", [None, 224, 224, 3], name="image_placeholder")
    ados = tf.compat.v1.placeholder("float", [None, 1, 50176,1], name="audio_placeholder")
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.compat.v1.Session(config=config) as sess:

        model = Resnet_18(imgs, ados, REG_PENALTY=0, is_training=False)
        pre_reader = tf.compat.v1.TFRecordReader()
        pre_filename_queue = tf.compat.v1.train.string_input_producer(
            ["predict/res/predict_res.tfrecords"], num_epochs=1
        )
        _, pre_serialized_example = pre_reader.read(pre_filename_queue)

        pre_feature =  {
            "predict/audio": tf.compat.v1.FixedLenFeature([], tf.string),
            "predict/image": tf.compat.v1.FixedLenFeature([], tf.string)
        }
        pre_features = tf.compat.v1.parse_single_example(
            pre_serialized_example, features=pre_feature
        )

        pre_image = tf.compat.v1.decode_raw(pre_features["predict/image"], tf.uint8)
        pre_audio = tf.compat.v1.decode_raw(pre_features["predict/audio"], tf.float32)
        pre_image = tf.reshape(pre_image, [224, 224, 3])
        pre_audio = tf.reshape(pre_audio, [1, 50176, 1])
        pre_images, pre_audios = tf.compat.v1.train.shuffle_batch(
            [pre_image, pre_audio],
            batch_size=BATCH_SIZE,
            capacity=50,
            min_after_dequeue=BATCH_SIZE,
            allow_smaller_final_batch=True,
        )
        init_op = tf.group(
            tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()
        )
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord)


        model.load_trained_model("param/param_resnet.pkl", sess)

        i = 0
        error = 0
        result = []
        while i < images_num:
            i += BATCH_SIZE
            try:
                epoch_x, epoch_y = sess.run([pre_images, pre_audios])
            except:
                if error >= 1:
                    break
                error += 1
                continue
            output = sess.run(
                [model.output], feed_dict={imgs: epoch_x.astype(np.float32), ados: epoch_y.astype(np.float32)}
            )
            result.append(output[0])

        coord.request_stop()
        coord.join(threads)

    # print(np.concatenate(result))
    output = np.mean(np.concatenate(result), axis=0)
    return output


def Res_result(video_name, value_1):
    output = predict(video_name)
    output = [round(i, 4) for i in output]
    value = np.array(value_1)
    output_json = {
        "Extraversion": output[0],
        "Neuroticism": output[1],
        "Agreeableness": output[2],
        "Conscientiousness": output[3],
        "Openness": output[4],
    }
    acc = 1 - np.absolute(output - value)
    acc = [round(i, 4) for i in acc]
    mean_acc = round(np.mean(acc),4)
    acc_json = {
        "Extraversion": acc[0],
        "Neuroticism": acc[1],
        "Agreeableness": acc[2],
        "Conscientiousness": acc[3],
        "Openness": acc[4],
    }
    print("The result of combine:\n", output_json)
    print("ACCURACY:\n", acc_json)
    print("MEAN_ACCURACY:\n", round(mean_acc, 4))
    return output, acc, mean_acc




