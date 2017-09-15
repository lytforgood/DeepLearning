
# coding: utf-8

# ## LSTM之keras实现

# In[1]:

import numpy as np
np.random.seed(2017)  #为了复现
from __future__ import print_function
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense
from keras.optimizers import Adam

(X_train, y_train), (X_test, y_test) = mnist.load_data()

#参数
#学习率
learning_rate = 0.001
#迭代次数
epochs = 2
#每块训练样本数
batch_size = 128
#输入
n_input = 28
#步长
n_step = 28
#LSTM Cell
n_hidden = 128
#类别
n_classes = 10

#x标准化到0-1  y使用one-hot  输入 nxm的矩阵 每行m维切成n个输入
X_train = X_train.reshape(-1, n_step, n_input)/255.
X_test = X_test.reshape(-1, n_step, n_input)/255.

y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)


# In[2]:

model = Sequential()
model.add(LSTM(n_hidden,
               batch_input_shape=(None, n_step, n_input),
               unroll=True))

model.add(Dense(n_classes))
model.add(Activation('softmax'))

adam = Adam(lr=learning_rate)
#显示模型细节
model.summary()
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1, #0不显示 1显示
          validation_data=(X_test, y_test))

scores = model.evaluate(X_test, y_test, verbose=0)
print('LSTM test score:', scores[0]) #loss
print('LSTM test accuracy:', scores[1])

