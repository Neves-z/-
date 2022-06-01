# Dan&& logfilterbank模型说明



### 文件夹

1. Annotations 保存标注
2. param  保存训练的参数 训练结束生成 
3. model_dan 保存模型 训练结束生成

### py文件

1. Dan_train.py 训练
2. dan_plus.py  图片的神经网络模型  
3. Logf_reg.py 音频线性回归器
4. Dan_write_into_tfrecords.py 把训练集和测试集写入tf文件 便于训练时读取
5. Dl_val_acc.py  计算训练好的模型的精度
6. Remtime.py 转换估计的剩余训练时间



### 其他

Dan_image_train.tfrecords 训练集 tf格式便于神经网络读取

Dan_image_val.tfrecords 测试集 

vgg-face.mat 训练好的vgg模型参数 用来初始化模型

loss_dan_image_full 保存loss的值 运行结束后生成

依赖库的话还是之前的那个



### 运行(和之前Resnet18模型基本步骤一样)

- 先把原始数据集（ImageData VoiceData）下载、解压之后放入文件夹

- 运行Dan_write_into_tfrecords.py 把训练集和测试集写入tf文件 便于训练时读取  

- 运行Dan_train.py  进行训练    

- 运行Dl_val_acc.py  计算训练好的模型的精度  


















