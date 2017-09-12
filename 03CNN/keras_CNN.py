
# coding: utf-8

# **CNN 实现**
#
# CNN相比与传统神经网络，主要区别是引入了卷积层和池化层
# 卷积是使用tf.nn.conv2d, 池化使用tf.nn.max_pool

# ## CNN之keras实现

# In[11]:

import numpy as np
np.random.seed(2017)  #为了复现
from __future__ import print_function
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

(X_train, y_train), (X_test, y_test) = mnist.load_data()

#x标准化到0-1  y使用one-hot
X_train = X_train.reshape(-1, 28,28, 1)/255.
X_test = X_test.reshape(-1, 28,28, 1)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)


# [卷积层Convolutional官方](http://keras-cn.readthedocs.io/en/latest/layers/convolutional_layer/)
#
# keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
#

# In[12]:

#建立模型 使用卷积层
model = Sequential()
#输出的维度 “same”代表保留边界处的卷积结果 “valid”代表只进行有效的卷积，即对边界数据不处理   height & width & channels
model.add(Conv2D(32, (5, 5),padding='same', activation='relu', input_shape=(28, 28, 1)))
#pool_size下采样因子 strides步长
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2),border_mode='same'))
#断开的神经元的比例
#model.add(Dropout(0.25))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(10, activation='softmax'))


# In[18]:

#定义优化器
adam = Adam(lr=1e-4)

#定义loss和评价函数 metrics评价可为cost，accuracy，score
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#训练模型 epoch训练次数 batch_size 每批处理32个
model.fit(X_train, y_train, epochs=1, batch_size=32)

#返回测试的指标
loss, accuracy = model.evaluate(X_test, y_test)
print('\n test loss: ', loss)
print('\n test accuracy: ', accuracy)

#预测
y_pre = model.predict(X_test)
#转换成数字-每列概率最大的位置
y_num=[np.argmax(x) for x in y_pre]

