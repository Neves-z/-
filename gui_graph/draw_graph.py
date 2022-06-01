import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from Dl_predict import Dl_result
from Dr_predict import Dr_result
from Res_predict import Res_result
from tkinter import messagebox

"""
def draw_graph():
    video_name = "8pM3X-xsD4s.005.mp4"
    new = [0.4580, 0.5417, 0.5385, 0.5048, 0.5000]
    value = np.array(new)
    Dl_output, Dl_acc, Dl_mean_acc = Dl_result(video_name,value)
    # Dr_output, Dr_acc, Dr_mean_acc = Dr_result(video_name, value)
    # Res_output, Res_acc, Res_mean_acc = Res_result(video_name, value)
    # plt.xticks(x)
    plt.bar(1, Dl_output[0], width=0.5, facecolor='#FFC0CB', label='Extraversion')
    plt.bar(2, Dl_output[1], width=0.5, facecolor='#6495ED', label='Neurotisicm')
    plt.bar(3, Dl_output[2], width=0.5, facecolor='#5F9EA0', label='agreeableness')
    plt.bar(4, Dl_output[3], width=0.5, facecolor='#FFDEAD', label='Conscientiousness')
    plt.bar(5, Dl_output[4], width=0.5, facecolor='#FF7F50', label='Openness')
    plt.legend(loc='upper left')
    plt.ylim([0, 1])
    plt.text(1, Dl_output[0], Dl_output[0], ha='center', va='bottom', fontsize=10)
    plt.text(2, Dl_output[1], Dl_output[1], ha='center', va='bottom', fontsize=10)
    plt.text(3, Dl_output[2], Dl_output[2], ha='center', va='bottom', fontsize=10)
    plt.text(4, Dl_output[3], Dl_output[3], ha='center', va='bottom', fontsize=10)
    plt.text(5, Dl_output[4], Dl_output[4], ha='center', va='bottom', fontsize=10)

    plt.xlabel('Apparent personality analysis', fontsize=13)

    plt.plot([1, 2, 3, 4, 5], Dl_acc, 'o-',color="r")

    plt.text(1, Dl_acc[0], Dl_acc[0], color="black", ha='center', va='bottom', fontsize=10)
    plt.text(2, Dl_acc[1], Dl_acc[1], color="black",ha='center', va='bottom', fontsize=10)
    plt.text(3, Dl_acc[2], Dl_acc[2], color="black",ha='center', va='bottom', fontsize=10)
    plt.text(4, Dl_acc[3], Dl_acc[3], color="black",ha='center', va='bottom', fontsize=10)
    plt.text(5, Dl_acc[4], Dl_acc[4], color="black",ha='center', va='bottom', fontsize=10)
    plt.show()
"""

def Video_value(pickle_file,Videoname,esist=False):
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
        for i in range(len(df)):
            if df.iat[i, 0] == Videoname:
                value = np.round(np.array(df.drop(["VideoName"], 1, inplace=False).iloc[i]),4)
                esist = True
                return value, esist

    print("该视频不存在")
    return None, esist
def draw(video_name):

    value, esist = Video_value("Annotations/annotation_validation.pkl",video_name,esist=False)
    print("ok")
    if(esist== False):
        return None
    messagebox.showinfo('提示', '正在预测中，请稍后！')
    Dl_output, Dl_acc, Dl_mean_acc = Dl_result(video_name,value)
    Res_output, Res_acc, Res_mean_acc = Res_result(video_name, value)
    Dr_output, Dr_acc, Dr_mean_acc = Dr_result(video_name, value)
    out = []
    out += [Dl_output]
    out += [Dr_output]
    out += [Res_output]



    acc = []
    acc += [Dl_acc]
    acc += [Dr_acc]
    acc += [Res_acc]

    mean_acc = []
    mean_acc += [Dl_mean_acc]
    mean_acc += [Dr_mean_acc]
    mean_acc += [Res_mean_acc]



    plt.figure(figsize=(10, 10), dpi=80)
    plt.figure(1)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签

    canve = [221,222,223]
    graph_name = ['DBR',
                  'DRN',
                  'DMR(my)',
                  ]
    for i in range(len(canve)):
        plt.subplot(canve[i])
        value = out[i]
        plt.bar(0.5, value[0], width=0.5, facecolor='#FFC0CB', label='Extraversion')
        plt.bar(1.5, value[1], width=0.5, facecolor='#6495ED', label='Neurotisicm')
        plt.bar(2.5, value[2], width=0.5, facecolor='#5F9EA0', label='agreeableness')
        plt.bar(3.5, value[3], width=0.5, facecolor='#FFDEAD', label='Conscientiousness')
        plt.bar(4.5, value[4], width=0.5, facecolor='#FF7F50', label='Openness')

        plt.ylim([0, 1.05])
        plt.text(0.5, value[0], value[0], ha='center', va='bottom', fontsize=10)
        plt.text(1.5, value[1], value[1], ha='center', va='bottom', fontsize=10)
        plt.text(2.5, value[2], value[2], ha='center', va='bottom', fontsize=10)
        plt.text(3.5, value[3], value[3], ha='center', va='bottom', fontsize=10)
        plt.text(4.5, value[4], value[4], ha='center', va='bottom', fontsize=10)

        plt.plot([0.5, 1.5, 2.5, 3.5, 4.5], acc[i], 'o-',color="r")
        Accuracy = acc[i]
        plt.text(0.5, Accuracy[0], Accuracy[0], color="black", ha='center', va='bottom', fontsize=10)
        plt.text(1.5, Accuracy[1], Accuracy[1], color="black",ha='center', va='bottom', fontsize=10)
        plt.text(2.5, Accuracy[2], Accuracy[2], color="black",ha='center', va='bottom', fontsize=10)
        plt.text(3.5, Accuracy[3], Accuracy[3], color="black",ha='center', va='bottom', fontsize=10)
        plt.text(4.5, Accuracy[4], Accuracy[4], color="black",ha='center', va='bottom', fontsize=10)

        # plt.legend(loc='upper left')
        plt.xlabel(graph_name[i], fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        x = [0.5, 1.5, 2.5, 3.5, 4.5]
        names = ['外倾性', '情绪稳定性', '随和性', '责任性', '开明性']
        plt.xticks(x, names, fontsize=10,rotation=15) #, rotation=30



    plt.subplot(224)
    plt.plot([0.5, 1.5, 2.5], mean_acc,'o-', color="blue")
    plt.text(0.5, mean_acc[0], mean_acc[0], color="black", ha='center', va='bottom', fontsize=10)
    plt.text(1.5, mean_acc[1], mean_acc[1], color="black", ha='center', va='bottom', fontsize=10)
    plt.text(2.5, mean_acc[2], mean_acc[2], color="black", ha='center', va='bottom', fontsize=10)

    names = ['DBR', 'DRN', 'DMR(MY)']

    x = [0.5, 1.5, 2.5]
    plt.xticks(x, names, fontsize=14)
    plt.xlabel('Mean accuracy', fontsize=14)
    plt.xlim([0, 3])
    plt.ylim([0.5, 1])

    plt.show()

# draw_graph()
#draw("MsGTYOOp4hE.001.mp4") ##XhgDsQlEnuU.005.mp4