
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def draw_graph():
    new = [0.8835, 0.8882, 0.8874, 0.8935, 0.9053, 0.9059, 0.8862, 0.8899]
    value = np.array(new)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    # plt.rcParams['axes.unicode_minus'] = True

    # plt.xticks(x)
    plt.bar(1, new[0], width=0.5, facecolor='#FFC0CB', label='DBR')
    plt.bar(1.5, new[1], width=0.5, facecolor='#6495ED', label='DMR')

    plt.bar(2.5, new[2], width=0.5, facecolor='#FFC0CB')
    plt.bar(3, new[3], width=0.5, facecolor='#6495ED')

    plt.bar(4, new[4], width=0.5, facecolor='#5F9EA0', label='DRN')
    plt.bar(4.5, new[5], width=0.5, facecolor='#6495ED')

    plt.bar(5.5, new[6], width=0.5, facecolor='#FFC0CB')
    plt.bar(6, new[7], width=0.5, facecolor='#6495ED')


    plt.legend(loc='upper left')
    plt.ylim([0.88, 0.909])
    plt.text(1, new[0], new[0], ha='center', va='bottom', fontsize=10)
    plt.text(1.5, new[1], new[1], ha='center', va='bottom', fontsize=10)
    plt.text(2.5, new[2], new[2], ha='center', va='bottom', fontsize=10)
    plt.text(3, new[3], new[3], ha='center', va='bottom', fontsize=10)
    plt.text(4, new[4], new[4], ha='center', va='bottom', fontsize=10)
    plt.text(4.5, new[5], new[5], ha='center', va='bottom', fontsize=10)
    plt.text(5.5, new[6], new[6], ha='center', va='bottom', fontsize=10)
    plt.text(6, new[7], new[7], ha='center', va='bottom', fontsize=10)



    #plt.xlabel("五大人格特征预测得分")  # 有中文出现的情况，需要u'内容'
    x = [1.25, 2.75, 4.25, 5.75]
    names = ['视觉单模态', '音频单模态', '视听早期融合', '视听晚期融合']
    plt.xticks(x, names, fontsize=10, rotation=15)  # , rotation=30
   # plt.xlabel('Apparent personality analysis', fontsize=13)

    plt.show()

draw_graph()