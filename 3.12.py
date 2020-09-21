# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         3.12
# Description:
# Author:       xuyapeng
# Date:         2020/9/1
# -------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras import layers, models, initializers, optimizers, regularizers
import numpy as np
import matplotlib.pyplot as plt


def linreg(X, w, b):
    return tf.matmul(X, w) + b


def squared_loss(y_true, y_hate):
    # tf.square(y_true - y_hate)
    # return tf.reduce_mean(tf.square(y_true, y_hate))
    return tf.square(y_true - y_hate) / 2


def sgd(params, lr, batch_size, grads):
    """Mini-batch stochastic gradient descent."""
    for i, param in enumerate(params):
        param.assign_sub(lr * grads[i] / batch_size)


def l2_penalty(w):
    return tf.reduce_sum((w ** 2)) / 2


n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = tf.ones((num_inputs, 1)) * 0.01, 0.05

features = tf.random.normal(shape=(n_train + n_test, num_inputs))
labels = tf.matmul(features, true_w) + true_b
labels += tf.random.normal(shape=labels.shape, mean=0.01, stddev=0.01)

train_features, test_features = features[:n_train], features[n_train:]
train_labels, test_labels = labels[:n_train], labels[n_train:]


def init_params():
    W = tf.Variable(tf.random.normal(mean=1, shape=(num_inputs, 1)))
    b = tf.Variable(tf.zeros(shape=(1,)))
    return [W, b]


batch_size = 1
num_epochs = 100
lr = 0.003
net = linreg
loss = squared_loss
train_iter = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).batch(batch_size).shuffle(batch_size)


def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                l = loss(y, net(X, w, b)) + lambd * l2_penalty(w)
            grads = tape.gradient(l, [w, b])
            sgd([w, b], lr, batch_size, grads)

        train_ls.append(tf.reduce_mean(loss(train_labels, net(train_features, w, b),
                                            )).numpy())
        test_ls.append(tf.reduce_mean(loss(test_labels, net(test_features, w, b),
                                           )).numpy())
        print('L2 norm of w:', tf.norm(w).numpy())
    print(train_ls)
    print(test_ls)


fit_and_plot(lambd=0)
