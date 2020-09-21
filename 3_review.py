import random

import tensorflow as tf

## 线性回归
num_samples = 10000
num_features = 3
num_outputs = 1

features = tf.random.normal(shape=(num_samples, num_features))
true_w = tf.reshape(tf.constant([2.4, 3.5, -2.0], dtype=tf.float32), shape=(num_features, num_outputs))
true_b = tf.constant([9.0], dtype=tf.float32, shape=(num_outputs,))

labels = tf.matmul(features, true_w) + true_b
## 增加噪声
labels += tf.random.normal(shape=labels.shape, stddev=0.01)

W = tf.Variable(tf.random.normal(shape=(num_features, num_outputs), stddev=0.01))
b = tf.Variable(tf.zeros(shape=(num_outputs,), dtype=tf.float32))


def data_iter(features, labels, batch_size):
    sample_num = len(features)
    index_list = list(range(sample_num))
    random.shuffle(index_list)
    for i in range(0, sample_num, batch_size):
        j = index_list[i: min(i + batch_size, sample_num)]
        yield tf.gather(features, indices=j), tf.gather(labels, indices=j)


def linreg(X, w, b):
    return tf.matmul(X, w) + b


def squared_loss(y_hat, y):
    return tf.square(y_hat - y) / 2.


def sgd(params, lr, batch_size, grads):
    """Mini-batch stochastic gradient descent."""
    for i, param in enumerate(params):
        param.assign_sub(lr * grads[i] / batch_size)


## test code
for X, y in data_iter(features, labels, batch_size=100):
    print(X.shape)
    print(y.shape)
    y_hat = linreg(X, W, b)
    loss = squared_loss(y_hat, y)
    print(loss.shape)
    break

num_epochs = 5
batch_size = 100
net = linreg
loss = squared_loss
lr = 0.01
for epoch in range(0, num_epochs + 1):
    for X, y in data_iter(features, labels, batch_size):
        with tf.GradientTape() as tapes:
            tapes.watch([W, b])
            y_hat = linreg(X, W, b)
            l = tf.reduce_sum(loss(y_hat, y))
        grads = tapes.gradient(l, [W, b])
        sgd([W, b], lr, batch_size, grads)
    labels_hat = linreg(features, W, b)
    print("epoch: {}   loss:{:.6f}".format(epoch, tf.reduce_mean(loss(labels_hat, labels))))

## 使用tensorflow自己搭建一个神经网络
