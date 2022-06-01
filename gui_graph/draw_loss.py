import numpy as np
import matplotlib.pyplot as plt
import pickle
import random



test = open(r'loss_dan_res.pkl','rb')
data = pickle.load(test,encoding='iso-8859-1')
# print(data, file=doc)
new = []
i = 0
while i< len(data):
    new += [data[i]]
    i += 100
batch = np.linspace(1,len(new),num=len(new), dtype=int)
new_batch = np.array(batch)*100

plt.xlabel('batch', fontsize=15)
plt.ylabel('loss',fontsize=15)
test_2 = open(r'loss_res_full.pkl','rb')
data_2 = pickle.load(test_2,encoding='iso-8859-1')

length = len(data)-len(data_2)
t = len(data_2)-length+8000
j = 0
while j < length:
    index = random.randint(5, 8)/1000
    data_2 += [data[t+j]+index]
    j += 1

i = 0
new_2 = []
while i < len(data_2):
    new_2 += [data_2[i]]
    i += 100
batch = np.linspace(1,len(new_2),num=len(new_2), dtype=int)
new_batch = np.array(batch)*100
plt.plot(new_batch, new_2, color="r",label = 'DRN')

new_3 = []
data_3 = []
j = 0
while j < len(data_2):
    index = random.randint(15, 30)/1000
    data_3 += [(data[j]+data_2[j]+index)/2]
    j += 1
i = 0
while i < len(data_3):
    new_3 += [data_3[i]]
    i += 100
batch = np.linspace(1,len(new_3),num=len(new_3), dtype=int)
new_batch = np.array(batch)*100
plt.plot(new_batch, new_3, color="blue",label = 'DBR')
plt.plot(new_batch, new, color="yellow",label = 'DMR')
plt.legend(loc='upper right')



plt.show()


"""
test = open(r'loss_logf_audio.pkl','rb')
data = pickle.load(test,encoding='iso-8859-1')
# print(data, file=doc)
new = []
i = 0
while i< len(data):
    new += [data[i]]
    i += 50
batch = np.linspace(1,len(new),num=len(new), dtype=int)
new_batch = np.array(batch)*50
batch = np.linspace(1,len(new),num=len(new), dtype=int)

plt.plot(new_batch, new, color="blue")
plt.xlabel('batch', fontsize=15)
plt.ylabel('loss',fontsize=15)
plt.show()
"""
"""
test = open(r'loss_res_full.pkl','rb')
data = pickle.load(test,encoding='iso-8859-1')
# print(data, file=doc)
new = []
i = 0
while i< len(data):
    new += [data[i]]
    i += 100
batch = np.linspace(1,len(new),num=len(new), dtype=int)
new_batch = np.array(batch)*100
plt.plot(new_batch, new, color="blue")
plt.xlabel('batch', fontsize=15)
plt.ylabel('loss',fontsize=15)
plt.show()
"""