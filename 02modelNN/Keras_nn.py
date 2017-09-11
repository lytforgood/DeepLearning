
# coding: utf-8

# ## Keras实现神经网络

# In[1]:

import numpy as np
np.random.seed(2017)  #为了复现
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop


# **数据格式说明**
# - x为28x28的矩阵(60000train+10000test)
# - y是0-9的数字

# In[3]:

(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[14]:

#x标准化到0-1  y使用one-hot
X_train = X_train.reshape(X_train.shape[0], -1) / 255.   # 把28x28 展开 -1是自动算列数 同时归一化 图像0-255
X_test = X_test.reshape(X_test.shape[0], -1) / 255.      # 把28x28 展开 -1是自动算列数 同时归一化 图像0-255
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)


# In[22]:

#建立模型 这里只使用了2层隐藏层
model = Sequential([
    Dense(32, input_dim=784),  #32 是输出的维度，784 是输入的维度
    Activation('relu'), #激励函数用到的是 relu 函数
    Dense(10),  #10个输出  输入不用定义 默认为上一层输出
    Activation('softmax'), #最后激励函数是 softmax
])

#定义优化器
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

#定义loss和评价函数 metrics评价可为cost，accuracy，score
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy', #crossentropy交叉熵
              metrics=['accuracy'])

#训练模型 epoch训练次数 batch_size 每批处理32个
model.fit(X_train, y_train, epochs=2, batch_size=32)

#返回测试的指标
loss, accuracy = model.evaluate(X_test, y_test)
print('test loss: ', loss)
print('test accuracy: ', accuracy)


# In[28]:

#预测
y_pre = model.predict(X_test)
#转换成数字-每列概率最大的位置
y_num=[np.argmax(x) for x in y_pre]

