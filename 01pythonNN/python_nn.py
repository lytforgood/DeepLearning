
# coding: utf-8

# ## MNIST数据集 
# **每张图像是28 * 28像素 手写数字**
# - train-images-idx3-ubyte  训练数据图像  (60,000)
# - train-labels-idx1-ubyte    训练数据label
# - t10k-images-idx3-ubyte   测试数据图像  (10,000)
# - t10k-labels-idx1-ubyte     测试数据label

# In[52]:

from __future__ import print_function
import numpy as np
import random

#初始化w b 输入为 [每层的size] eg: [4,5,2] 输入层为4 隐藏层为 5 输出层为 2
def initwb(sizes):
    num_layers_ = len(sizes)  #层数
    w_ = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] #1-最后二层 与 2-最后一层 用zip索引形成元组索引 生成后一层x前一层的矩阵
    b_ = [np.random.randn(y, 1) for y in sizes[1:]]  # w_、b_初始化为正态分布随机数
    return w_ ,b_,num_layers_


# In[53]:

# Sigmoid函数，S型曲线，
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


# In[54]:

# Sigmoid函数的导函数
def sigmoid_prime(z):
#     return (1.0/(1.0+np.exp(-z)))/(1+1.0/(1.0+np.exp(-z)))
    return sigmoid(z)/(1+sigmoid(z))


# In[55]:

#定义前馈(feedforward)函数 给神经网络的输入x，输出对应的值
def feedforward(w_,b_,x):
    for b, w in zip(b_, w_): ##前向传播 每层进行计算 zip把每层的w b给选择出来
        x = sigmoid(np.dot(w, x)+b) ##计算每层的  w*输入+b
    return x


# In[56]:

##计算损失函数倒数
def cost_derivative(output_activations, y):
    return (output_activations-y)


# In[57]:

##反向传播
def backprop(x, y,w_,b_,num_layers_):
    nabla_b = [np.zeros(b.shape) for b in b_]
    nabla_w = [np.zeros(w.shape) for w in w_]
    #激活函数输入
    activation = x
    activations = [x]
    zs = []
    for b, w in zip(b_, w_):
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)
 
    delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose()) ##transpose转置
 
    for l in range(2, num_layers_):
        z = zs[-l]
        sp = sigmoid_prime(z)
        delta = np.dot(w_[-l+1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return (nabla_b, nabla_w)


# In[58]:

##更新每个块 进行参数训练
def update_mini_batch(mini_batch, eta,w_,b_,num_layers_):
    nabla_b = [np.zeros(b.shape) for b in b_]
    nabla_w = [np.zeros(w.shape) for w in w_]
    for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = backprop(x, y,w_,b_,num_layers_)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        w_ = [w-(eta/len(mini_batch))*nw for w, nw in zip(w_, nabla_w)]
        b_ = [b-(eta/len(mini_batch))*nb for b, nb in zip(b_, nabla_b)]
    return  w_,b_


# In[59]:

def evaluate(test_data,w_,b_):
    test_results = [(np.argmax(feedforward(w_,b_,x)), y) for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)


# In[60]:

#随机梯度下降 training_data是训练数据(x, y); epochs是训练次数; mini_batch_size是每次训练样本数; eta是learning rate
def SGD(training_data, epochs, mini_batch_size, eta, test_data=None,w_=None, b_=None, num_layers_=None):
    if test_data:
        n_test = len(test_data)
    n = len(training_data)
    for j in range(epochs):
        random.shuffle(training_data) #打乱顺序
        mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)] #生成不同块
        for mini_batch in mini_batches:
            w_,b_=update_mini_batch(mini_batch, eta,w_,b_,num_layers_)
        if test_data:
            print("Epoch {0}: {1} / {2}".format(j, evaluate(test_data,w_,b_), n_test)) ##{索引} format 索引值
        else:
            print("Epoch {0} complete".format(j)) 
    return  w_,b_     


# In[61]:

##预测
def predict(data,w_,b_):
    value = feedforward(w_,b_,data)
    return value.tolist().index(max(value))


# In[37]:

##处理数据
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
 
def load_mnist(dataset="training_data", digits=np.arange(10), path="./MNIST_data/"):
 
    if dataset == "training_data":
        fname_image = os.path.join(path, 'train-images-idx3-ubyte')
        fname_label = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing_data":
        fname_image = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_label = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'training_data' or 'testing_data'")
 
    flbl = open(fname_label, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()
 
    fimg = open(fname_image, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()
 
    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)
 
    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]
 
    return images, labels
 
def load_samples(dataset="training_data"):
    image,label = load_mnist(dataset)
    #print(image[0].shape, image.shape)   # (28, 28) (60000, 28, 28)
    #print(label[0].shape, label.shape)   # (1,) (60000, 1)
    #print(label[0])   # 5
 
    # 把28*28二维数据转为一维数据
    X = [np.reshape(x,(28*28, 1)) for x in image]
    X = [x/255.0 for x in X]   # 灰度值范围(0-255)，转换为(0-1)
    #print(X.shape)
 
    # 5 -> [0,0,0,0,0,1.0,0,0,0]      1 -> [0,1.0,0,0,0,0,0,0,0]
    def vectorized_Y(y):
        e = np.zeros((10, 1))
        e[y] = 1.0
        return e
    # 把Y值转换为神经网络的输出格式
    if dataset == "training_data":
        Y = [vectorized_Y(y) for y in label]
        pair = list(zip(X, Y))
        return pair
    elif dataset == 'testing_data':
        pair = list(zip(X, label))
        return pair
    else:
        print('Something wrong')


# In[38]:

##定义 输入 输出 大小
INPUT = 28*28
OUTPUT = 10
##提取数据
train_set = load_samples(dataset='training_data')
test_set = load_samples(dataset='testing_data') ## 每一个样本是 28*28=784x1 + label


# In[62]:

##初始化权重
w_,b_,num_layers_=initwb([INPUT, 36, OUTPUT]) 
new_w,new_b=SGD(train_set, 10, 100, 1.0, test_data=test_set,w_=w_,b_=b_,num_layers_=num_layers_)


# In[64]:

#准确率
correct = 0;
for test_feature in test_set:
    if predict(test_feature[0],new_w,new_b) == test_feature[1][0]:
        correct += 1
print("准确率: ", float(correct)/float(len(test_set)))

