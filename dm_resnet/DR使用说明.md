# Dan_max&resnet34使用说明



### 文件夹

1. Annotations 保存标注
2. param  保存训练的参数 训练结束生成 
3. model_dan_res保存模型 训练结束生成

### py文件

1. Dr_train.py 训练
2. Dr_net.py  图片的神经网络模型  图片部分使用Vgg19改编 音频使用Resnet34
3. Dr_data_to_tf.py 把训练集和测试集写入tf文件 便于训练时读取
4. Dr_val_acc.py  计算训练好的模型的精度
5. Remtime.py 转换估计的剩余训练时间



### 其他

Dr_train.tfrecords 训练集 tf格式便于神经网络读取

Dr_val.tfrecords 测试集 

loss_dan_res_full.pkl  保存loss的值 运行结束后生成

依赖库的话还是之前的那个



### 运行

- 先把原始数据集（ImageData VoiceData）下载、解压之后放入文件夹 

- 运行Dr_data_to_tf.py 把训练集和测试集写入tf文件 便于训练时读取  

- 运行Dr_train.py  进行训练   

- 运行Dr_val_acc.py  计算训练好的模型的精度  


















