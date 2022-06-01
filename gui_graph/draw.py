
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def draw_graph():
    new = [0.4580, 0.5417, 0.5385, 0.5048, 0.5000]
    value = np.array(new)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    # plt.rcParams['axes.unicode_minus'] = True

    # plt.xticks(x)
    plt.bar(1, new[0], width=0.5, facecolor='#FFC0CB', label='随和性')
    plt.bar(2, new[1], width=0.5, facecolor='#6495ED', label='外倾性')
    plt.bar(3, new[2], width=0.5, facecolor='#5F9EA0', label='情绪稳定性')
    plt.bar(4, new[3], width=0.5, facecolor='#FFDEAD', label='责任性')
    plt.bar(5, new[4], width=0.5, facecolor='#FF7F50', label='开放性')
    plt.legend(loc='upper left')
    plt.ylim([0, 1])
    plt.text(1, new[0], new[0], ha='center', va='bottom', fontsize=10)
    plt.text(2, new[1], new[1], ha='center', va='bottom', fontsize=10)
    plt.text(3, new[2], new[2], ha='center', va='bottom', fontsize=10)
    plt.text(4, new[3], new[3], ha='center', va='bottom', fontsize=10)
    plt.text(5, new[4], new[4], ha='center', va='bottom', fontsize=10)


    #plt.xlabel("五大人格特征预测得分")  # 有中文出现的情况，需要u'内容'
    x = [0.5, 1.5, 2.5, 3.5, 4.5]
    names = ['外倾性', '情绪稳定性', '随和性', '责任性', '开放性']
    # plt.xticks(x, names, fontsize=10, rotation=15)  # , rotation=30
   # plt.xlabel('Apparent personality analysis', fontsize=13)

    plt.show()

draw_graph()