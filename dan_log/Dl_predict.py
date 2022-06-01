import glob
import os
import sys
import cv2
import warnings
import subprocess

import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav


from PIL import Image
from dan_plus import DAN_PLUS
from Logf_reg import Audio_reg
from python_speech_features import logfbank
tf.compat.v1.disable_eager_execution()


warnings.filterwarnings("ignore")

REG_PENALTY = 0


def load_image(addr):
    img = np.array(Image.open(addr).resize((224, 224), Image.ANTIALIAS))
    img = img.astype(np.uint8)
    return img

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


def predict_image(file_name):

    num = []

    cap = cv2.VideoCapture(file_name)

    file_name = (file_name.split(".mp4"))[0]
    ## Creating folder to save all the 100 frames from the video
    try:
        os.makedirs("ImageData/testingData/" + file_name)
    except OSError:
        print("Error: Creating directory of data")

    ## Setting the frame limit to 100
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

        ## Resizing it to 256*256 to save the disk space and fit into the model
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_CUBIC)
        # Saves image of the current frame in jpg file
        name = (
            "ImageData/testingData/" + str(file_name) + "/frame" + str(count) + ".jpg"
        )
        cv2.imwrite(name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break




    image_addrs = []

    filelist = glob.glob("ImageData/testingData/" + str(file_name) + "/*.jpg")
    image_addrs += filelist
    try:
        os.makedirs("predict/dan_log")
    except OSError:
        print("Error: Creating predict/dan_log")

    predict_filename = "predict/dan_log/predict_image.tfrecords"  # address to save the TFRecords file

    writer = tf.compat.v1.python_io.TFRecordWriter(predict_filename)
    for i in range(len(image_addrs)):
        # Load the image
        img = load_image(image_addrs[i])
        feature = {"predict/image": _bytes_feature(tf.compat.as_bytes(img.tobytes()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

    BATCH_SIZE = 25
    NUM_IMAGES = 100

    imgs = tf.compat.v1.placeholder("float", [None, 224, 224, 3], name="image_placeholder")
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.compat.v1.Session(config=config) as sess:

        model = DAN_PLUS(imgs, REG_PENALTY=REG_PENALTY, preprocess="vggface")
        pre_reader = tf.compat.v1.TFRecordReader()
        pre_filename_queue = tf.compat.v1.train.string_input_producer(
            ["predict/dan_log/predict_image.tfrecords"], num_epochs=1
        )
        _, pre_serialized_example = pre_reader.read(pre_filename_queue)
        pre_feature = {"predict/image": tf.compat.v1.FixedLenFeature([], tf.string)}
        pre_features = tf.compat.v1.parse_single_example(
            pre_serialized_example, features=pre_feature
        )

        pre_image = tf.compat.v1.decode_raw(pre_features["predict/image"], tf.uint8)
        pre_image = tf.reshape(pre_image, [224, 224, 3])
        pre_images = tf.compat.v1.train.shuffle_batch(
            [pre_image],
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
        file_list = ["param/param_dl_imgs.pkl"]
        for pickle_file in file_list:
            error = 0
            model.load_trained_model(pickle_file, sess)
            i = 0
            while i < NUM_IMAGES:
                i += BATCH_SIZE
                try:
                    epoch_x = sess.run(pre_images)
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
    # print(np.concatenate(num))
    a = np.mean(np.concatenate(num), axis=0)
    return a


def predict_audio(file_name):
    file_name = (file_name.split(".mp4"))[0]
    try:
        if not os.path.exists("VoiceData/testingData/"):
            os.makedirs("VoiceData/testingData/")
    except OSError:
        print("Error: Creating directory of data")
    command = "/usr/local/bin/ffmpeg -i {}.mp4 -ab 320k -ac 2 -ar 44100 -vn VoiceData/testingData/{}.wav".format(
         file_name, file_name
    )
    subprocess.call(command, shell=True)



    audio_name = glob.glob("VoiceData/testingData/" + str(file_name) + ".wav")

    try:
        os.makedirs("predict/dan_log")
    except OSError:
        print("Error: Creating predict/dan_log")

    predict_filename = "predict/dan_log/predict_audio.tfrecords"  # address to save the TFRecords file

    writer = tf.compat.v1.python_io.TFRecordWriter(predict_filename)
    audio = process_wav(audio_name[0])
    if audio.shape != (1, 79534):
        print("The shape of test audio is not (1, 79534)! ")
        return None

    # Create a feature
    feature = {
        "predict/audio": _bytes_feature(tf.compat.as_bytes(audio.tobytes())),
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
        pre_reader = tf.compat.v1.TFRecordReader()
        pre_filename_queue = tf.compat.v1.train.string_input_producer(
            ["predict/dan_log/predict_audio.tfrecords"], num_epochs=1
        )
        _, pre_serialized_example = pre_reader.read(pre_filename_queue)
        pre_feature = {"predict/audio": tf.compat.v1.FixedLenFeature([], tf.string)}
        pre_features = tf.compat.v1.parse_single_example(pre_serialized_example, features=pre_feature)
        pre_audio = tf.compat.v1.decode_raw(pre_features["predict/audio"], tf.compat.v1.float64)
        # Reshape image data into the original shape
        pre_audio = tf.compat.v1.reshape(pre_audio, [1, 79534])
        init_op = tf.group(
            tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()
        )
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord)



        file_list = ["param/param_dl_ados.pkl"]
        for pickle_file in file_list:
            model.load_trained_model(pickle_file, sess)
            try:
                epoch_x = sess.run(pre_audio)
            except:
                print("Error")
                return
            output = sess.run(
                [model.output], feed_dict={Ados: epoch_x.astype(np.float64)}
            )

        coord.request_stop()
        coord.join(threads)
    a = output[0]
    return a[0]

def Dl_result(video_name, value):
    output_image = predict_image(video_name)
    output_audio = predict_audio(video_name)
    '''
    output_image_json = {
        "Extraversion": output_image[0],
        "Neuroticism": output_image[1],
        "Agreeableness": output_image[2],
        "Conscientiousness": output_image[3],
        "Openness": output_image[4],
    }
    output_audio_json = {
        "Extraversion": output_audio[0],
        "Neuroticism": output_audio[1],
        "Agreeableness": output_audio[2],
        "Conscientiousness": output_audio[3],
        "Openness": output_audio[4],
    }
    print("The result of image:")
    print(output_image_json)
    print("The result of audio:")
    print(output_audio_json)
    '''
    output = []
    output.append(output_audio)
    output.append(output_image)
    out = np.round(np.mean(output, axis=0), 4)
    # new = [0.4580, 0.5417, 0.5385, 0.5048, 0.5000]
    # value = np.array(new)
    acc = 1 - np.absolute(out - value)
    acc = [np.round(i, 4) for i in acc]
    mean_acc = np.round(np.mean(acc),4)
    out_json = {
        "Extraversion": out[0],
        "Neuroticism": out[1],
        "Agreeableness": out[2],
        "Conscientiousness": out[3],
        "Openness": out[4],
    }
    acc_json = {
        "Extraversion": acc[0],
        "Neuroticism": acc[1],
        "Agreeableness": acc[2],
        "Conscientiousness": acc[3],
        "Openness": acc[4],
    }
    print("\nThe result of combine:\n", out_json)
    print("ACCURACY:\n", acc_json)
    print("MEAN_ACCURACY:\n", round(mean_acc, 4))
    return out, acc, mean_acc








