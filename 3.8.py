# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         3.8
# Description:  多层感知机章节
# Author:       xuyapeng
# Date:         2020/8/26
# -------------------------------------------------------------------------------

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import random


def set_figsize(figsize=(3.5, 2.5)):
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


def xyplot(x_vals, y_vals, name):
    plt.figure()
    set_figsize(figsize=(5, 2.5))
    plt.plot(x_vals.numpy(), y_vals.numpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')


x = tf.Variable(tf.range(-8, 8, 0.1), dtype=tf.float32)

with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    relu_y = tf.nn.relu(x)
    sigmoid_y = tf.nn.sigmoid(x)
    tanh_y = tf.nn.tanh(x)
relu_dy_dx = tape.gradient(relu_y, x)
sigmoid_dy_dx = tape.gradient(sigmoid_y, x)
tanh_dy_dx = tape.gradient(tanh_y, x)

row_num = 3
col_num = 2
fig, ax = plt.subplots(row_num, col_num, figsize=(10, 10))
fig.subplots_adjust(hspace=1, wspace=0.3)
plot_data_list = [{'relu': relu_y, 'relu_dy_dx': relu_dy_dx},
                  {'sigmoid': sigmoid_y, 'sigmoid_dy_dx': sigmoid_dy_dx},
                  {'tanh': tanh_y, 'tanh_dy_dx': tanh_dy_dx}]
# for i in range(row_num):
#     ax[i, 0].plot(x.numpy(), relu_y.numpy())
#     for j in range(1, col_num):
#         ax[i, j].plot(x.numpy(), relu_dy_dx.numpy())
for i in range(len(plot_data_list)):
    for j, key in enumerate(plot_data_list[i].keys()):
        ax[i, j].plot(x.numpy(), plot_data_list[i][key].numpy())
        ax[i, j].set_xlabel(key)
plt.show()
