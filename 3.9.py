# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         3.9
# Description:  多层感知机从零实现
# Author:       xuyapeng
# Date:         2020/8/27
# -------------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
from functools import reduce
from tensorflow.keras.datasets import fashion_mnist


class Model:
    def __init__(self, config, train_iter, test_iter):
        self.train_iter = train_iter
        self.test_iter = test_iter
        W1_shape = config['W1']
        b1_shape = config['b1']
        W2_shape = config['W2']
        b2_shape = config['b2']
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        initializer = tf.random.truncated_normal
        self.layer1_W = tf.Variable(initial_value=initializer(shape=W1_shape, stddev=0.01))
        self.layer1_b = tf.Variable(initial_value=initializer(shape=b1_shape, stddev=0.01))
        self.layer2_W = tf.Variable(initial_value=initializer(shape=W2_shape, stddev=0.01))
        self.layer2_b = tf.Variable(initial_value=initializer(shape=b2_shape, stddev=0.01))
        self.params = [self.layer1_W, self.layer1_b, self.layer2_W, self.layer2_b]

        self.model = self.net

    def net(self, X):
        reshaped_x = tf.cast(tf.reshape(X, (-1, self.layer1_W.shape[0])), dtype=tf.float32)
        output1 = self.relu(tf.matmul(reshaped_x, self.layer1_W) + self.layer1_b)
        output2 = self.relu(tf.matmul(output1, self.layer2_W) + self.layer2_b)
        return self.softmax(output2)

    @staticmethod
    def relu(x):
        return tf.math.maximum(x, 0)

    @staticmethod
    def softmax(x):
        return tf.exp(x) / tf.reduce_sum(tf.exp(x), axis=1, keepdims=True)

    def crossentropy(self, y_true, y_hat):
        y_true = tf.one_hot(y_true, depth=y_hat.shape[-1])
        return -y_true * tf.math.log(y_hat + 1e-8)

    def cross_entropy_2(self, y, y_hat):
        '''
        官方实现方式
        '''
        y = tf.cast(tf.reshape(y, shape=[-1, 1]), dtype=tf.int32)
        y = tf.one_hot(y, depth=y_hat.shape[-1])
        y = tf.cast(tf.reshape(y, shape=[-1, y_hat.shape[-1]]), dtype=tf.int32)
        return -tf.math.log(tf.boolean_mask(y_hat, y) + 1e-8)

    def accuracy(self, y_true, y_hat):
        predict = tf.argmax(y_hat, axis=1)
        y_true = tf.cast(y_true, dtype=predict.dtype)
        return np.sum(tf.equal(y_true, predict))

    def sgd(self):
        """Mini-batch stochastic gradient descent."""
        for i, parames in enumerate(self.params):
            parames.assign_sub(self.lr * self.grads[i] / self.batch_size)

    def evaluate_accuracy(self, data_iter):
        acc_sum, loss, n = 0.0, 0.0, 0
        for (batch_id, (X, y)) in enumerate(data_iter):
            y_hat = self.model(X)
            loss += tf.reduce_sum(self.crossentropy(y, y_hat))
            acc_sum += self.accuracy(y, y_hat)
            n += X.shape[0]
        return loss / n, acc_sum / n

    def train(self):
        train_loss, train_acc, n = 0.0, 0.0, 0
        for (batch_id, (batch_x, batch_y)) in enumerate(self.train_iter):
            with tf.GradientTape() as tape:
                tape.watch(self.params)
                logits = self.model(batch_x)
                loss = tf.reduce_sum(self.crossentropy(batch_y, logits))
            self.grads = tape.gradient(loss, self.params)
            # 更新权重
            self.sgd()
            train_loss += loss
            n += batch_x.shape[0]
            train_acc += self.accuracy(batch_y, logits)
            print("\rtrain loss: {:.4f} train acc: {:.4f}".format(train_loss / n, train_acc / n), end="")
        print()
        test_loss, test_acc = self.evaluate_accuracy(self.test_iter)
        print("train loss: {:.4f} train acc: {:.4f} | test loss: {:.4f} test acc {:.4f}".format(train_loss / n,
                                                                                                train_acc / n,
                                                                                                test_loss, test_acc
                                                                                                ))

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
batch_size = 256
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)
x_train = x_train / 255.0
x_test = x_test / 255.0
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
input_shape = None
for X, y in train_iter:
    input_shape = X.shape
    break


num_epoch = 10
lr = 0.1

num_inputs, num_outputs, num_hiddens = 784, 10, 256
if num_inputs is not None:
    num_inputs = reduce(lambda x, y: x * y, input_shape[1:])
net_config = {'W1': (num_inputs, num_hiddens),
              'b1': (num_hiddens,),
              'W2': (num_hiddens, num_outputs),
              'b2': (num_outputs,),
              'lr': lr,
              'batch_size': batch_size}

trainer = Model(net_config, train_iter, test_iter)
for epoch in range(num_epoch):
    print("============EPOCH: {}/{}===========".format(epoch + 1, num_epoch))
    trainer.train()
