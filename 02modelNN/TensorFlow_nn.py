
# coding: utf-8

# ## TensorFlow实现神经网络

# In[1]:

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# **数据格式说明**
# - x为nx784的矩阵(55000train+10000test)
# - y是nx10的矩阵(one-hot)

# In[2]:

X_train,y_train = mnist.train.images , mnist.train.labels
X_test,y_test = mnist.test.images , mnist.test.labels


# In[3]:

#定义计算准确率
def compute_accuracy(x_val, y_val):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: x_val})
    #tf.argmax(input,axis) 0表示按列，1表示按行
    #tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y_val,1)) #每行最大值所在索引
    #tf.cast类型转换 tf.reduce_mean 求平均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: x_val, ys: y_val})
    return result


# In[4]:

##产生随机变量
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# In[5]:

#定义输入占位符
xs = tf.placeholder(tf.float32, [None, 784]) # 行x列 维度
ys = tf.placeholder(tf.float32, [None, 10])

#添加隐藏层
W_h1 = weight_variable([784,32])
b_h1 = bias_variable([32])
x_h1 = tf.nn.relu(tf.matmul(xs, W_h1) + b_h1)

W_h2 = weight_variable([32, 10])
b_h2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(x_h1, W_h2) + b_h2)


#定义loss函数 交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
#定义训练优化
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


#定义会话
sess = tf.Session()
#初始化
init = tf.global_variables_initializer()

sess.run(init)


# In[7]:

##训练  sess相当于model 里面有参数w,b
batch_size = 100
n_chunk = len(X_train) // batch_size
for i in range(n_chunk):
    start_index = i * batch_size
    end_index = start_index + batch_size
    batch_xs,batch_ys = X_train[start_index:end_index] , y_train[start_index:end_index]
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(X_test,y_test))


# In[ ]:



