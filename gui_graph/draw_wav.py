import scipy.io.wavfile
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import random
import numpy as np

plt.figure(1,figsize=(7, 4.5))#8，5分别对应宽和高
plt.rcParams['axes.unicode_minus'] = False
(rate, sig) = scipy.io.wavfile.read("_0bg1TLPP-I.004.wav")
# 左声道
left = sig[:,0]/10000
# 右声道
# right = sig[:,1]
time = np.linspace(1,674816,num=674816, dtype=int)  #

def formatnum(x, pos):
    return '$%d$x$10^{5}$' % (x/1e5) # '$%.1f$x$10^{5}$

formatter = FuncFormatter(formatnum)

plt.gca().xaxis.set_major_formatter(formatter)

plt.xlabel('Time', fontsize=10)
plt.ylabel('Amplitude',fontsize=10)
plt.plot(time,left)

plt.xlim([0,700000])
plt.ylim([-3,3])

# plt.plot(right)
plt.show()

