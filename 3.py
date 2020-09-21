# 3.1  线性回归
## 3.1.1 线性回归的基本要素
## 模型  训练数据  损失函数
## 损失函数最小值解可以直接用公式表达出来，这类解叫做解析解
## 大多数深度学习模型并没有解析解，只能通过优化算法迭代模型参数尽可能降低损失函数值，这类解叫作数值解

## 3.1.2 线性回归的表示方法
## 全连接层定义：输出层中的神经元和输入层中各个输入完全连接。

import tensorflow as tf
from time import time
from tqdm import tqdm

a = tf.ones((1000,))
b = tf.ones((1000,))
c = tf.Variable(tf.zeros_like(a))
id_before = id(c)
c.assign(a + b)
print(id(c) == id_before)

# 3.2 线性回归从零开始实现
import tensorflow as tf
from matplotlib import pyplot as plt
import random

num_inputs = 2
num_examples = 1000
true_w = tf.constant([[2], [-3.4]])
true_b = 4.2
features = tf.random.normal((num_examples, num_inputs))
# labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
labels = tf.matmul(features, true_w) + true_b
labels += tf.random.normal(labels.shape, stddev=0.01)
print(features[0], labels[0])


def set_figsize(figsize=(3.5, 2.5)):
    plt.rcParams['figure.figsize'] = figsize


set_figsize()
plt.scatter(features[:, 1], labels[:, 0], 1)
plt.show()


## 读取数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = indices[i: min(i + batch_size, num_examples)]
        yield tf.gather(features, axis=0, indices=j), tf.gather(labels, axis=0, indices=j)


batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X)
    print(y)
    break

## 初始化权重
w = tf.Variable(tf.random.normal((num_inputs, 1), stddev=0.01))
b = tf.Variable(tf.zeros((1,)))


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


lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with tf.GradientTape() as t:
            t.watch([w, b])  # 这一条语句也不用写，默认会自动追踪Variable类型变量
            l = tf.reduce_sum(loss(y, net(X, w, b)))
        grads = t.gradient(l, [w, b])
        sgd([w, b], lr, batch_size, grads)
        # with tf.GradientTape() as t:
        #     l = tf.reduce_sum(loss(y, net(X, w, b)))
        #     grads = t.gradient(l, net.trainable_variables)
        # print(net.trainable_variables)
        # break
        # # sgd(net.trainable_variables)
    train_l = loss(labels, net(features, w, b))
    print('epoch %d, loss %f' % (epoch + 1, tf.reduce_mean(train_l)))
print(w)
print(b)

# 3.3 线性回归的简洁实现
import tensorflow as tf
from matplotlib import pyplot as plt
import random

num_inputs = 2
num_examples = 1000
true_w = tf.constant([[2], [-3.4]])
true_b = 4.2
features = tf.random.normal((num_examples, num_inputs))
# labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
labels = tf.matmul(features, true_w) + true_b
labels += tf.random.normal(labels.shape, stddev=0.01)

## 3.3.2 读取数据
from tensorflow import data as tfdata

batch_size = 10
# 将训练数据的特征和标签组合
dataset = tfdata.Dataset.from_tensor_slices((features, labels))
# 随机读取小批量
dataset = dataset.shuffle(buffer_size=num_examples)
dataset = dataset.batch(batch_size)
# data_iter = iter(dataset)
# for X, y in data_iter:
#     print(X)
#     print(y)
#     break  
for (batch, (X, y)) in enumerate(dataset):
    print(batch)
    print(X)
    print(y)
    break

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import initializers as init

model = keras.Sequential()
model.add(layers.Dense(1, kernel_initializer=init.RandomNormal(stddev=0.01)))

## 定义损失函数
from tensorflow import losses

loss = losses.MeanSquaredError()
## 定义优化算法
from tensorflow.keras import optimizers

trainer = optimizers.SGD(learning_rate=0.03)

num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for (batch, (X, y)) in enumerate(dataset):
        with tf.GradientTape() as tape:
            l = loss(model(X, training=True), y)
        grads = tape.gradient(l, model.trainable_variables)
        trainer.apply_gradients(zip(grads, model.trainable_variables))
    train_l = loss(model(features), labels)
    print('epoch %d, loss: %f' % (epoch, l))

print(model.trainable_variables)

# 3.4 softmax回归
## 交叉熵损失函数定义  最大似然估计概念

# 3.5 图像分类数据集
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import sys
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print("训练集个数：%d  验证集个数：%d" % (len(x_train), len(x_test)))


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    _, figs = plt.subplots(1, len(images), figsize=(10, 10))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)))
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


X, y = [], []
for i in range(10):
    X.append(x_train[i])
    y.append(y_train[i])
show_fashion_mnist(X, get_fashion_mnist_labels(y))

# 读取小批量
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))

# 3.6 softmax回归的从零开始
from tensorflow.keras.datasets import fashion_mnist

batch_size = 256
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = tf.cast(x_train, tf.float32) / 255  # 在进行矩阵相乘时需要float型，故强制类型转换为float型
x_test = tf.cast(x_test, tf.float32) / 255  # 在进行矩阵相乘时需要float型，故强制类型转换为float型
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=len(x_train)).batch(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(buffer_size=len(x_test)).batch(batch_size)

train_num = x_train.shape[0]
## 3.6.2 初始化模型参数
num_inputs = 784
num_outputs = 10
W = tf.Variable(tf.random.normal((num_inputs, num_outputs), stddev=0.01))
b = tf.Variable(tf.zeros(shape=(num_outputs,), dtype=tf.float32))


def softmax(logits, axis=-1):
    return tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis, keepdims=True)


def net(X, w, b):
    logits = tf.matmul(tf.reshape(X, shape=(-1, w.shape[0])), w) + b
    return softmax(logits)


def cross_entropy_2(y_hat, y):
    '''
    官方实现方式
    '''
    y = tf.cast(tf.reshape(y, shape=[-1, 1]), dtype=tf.int32)
    y = tf.one_hot(y, depth=y_hat.shape[-1])
    y = tf.cast(tf.reshape(y, shape=[-1, y_hat.shape[-1]]), dtype=tf.int32)
    return -tf.math.log(tf.boolean_mask(y_hat, y) + 1e-8)


def cross_entropy(y_hat, y):
    y = tf.one_hot(y, depth=y_hat.shape[-1])
    y = tf.cast(y, dtype=tf.float32)
    assert y.shape[0] == y_hat.shape[0] and y.shape[1] == y_hat.shape[1], "shape not equal error!"
    result = y * -tf.math.log(y_hat + 1e-8)
    return tf.reduce_sum(result, axis=1)  # 按行求和、求取每一个样本的和


def sgd(params, lr, batch_size, grads):
    """Mini-batch stochastic gradient descent."""
    for i, param in enumerate(params):
        param.assign_sub(lr * grads[i] / batch_size)


def accuracy(y_hat, y):
    y = tf.cast(y, dtype=tf.int64)
    return np.sum(tf.argmax(y_hat, axis=1) == y)


def evaluate_accuracy(data_iter, net, w, b):
    acc_sum, loss, n = 0.0, 0.0, 0
    for (batch_id, (X, y)) in enumerate(data_iter):
        y_hat = net(X, w, b)
        loss += tf.reduce_sum(cross_entropy(y_hat, y))
        acc_sum += accuracy(y_hat, y)
        n += X.shape[0]
    return loss / n, acc_sum / n


num_epochs = 5
lr = 0.1
model = net

# def train_ch3(model, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, trainer=None):
#     for epoch in range(num_epochs):
#         train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
#         for (batch_id, (X, y)) in train_iter:
#             with tf.GradientTape() as tape:
#                 tape.watch[params]
#                 y_hat = model(X, params[0], params[1])
#                 l = tf.reduce_sum(loss(y_hat, y))
#             grads = taps.gradient(l, params)
#
#             if trainer is None:
#                 for i, param in enumerate(params):
#                     param.assign_sub(lr * grads[i] / batch_size)
#             else:
#                 trainer.apply_gradients(zip([grad / batch_size for grad in grads], params))
#

for epoch in range(1, num_epochs + 1):
    train_l_sum, train_acc_num, n = 0.0, 0.0, 0
    for (bach_id, (X, y)) in enumerate(train_iter):
        with tf.GradientTape() as taps:
            taps.watch([W, b])
            y_hat = model(X, W, b)
            print(tf.reduce_sum(y_hat, axis=1))
            loss = cross_entropy(y_hat, y)  # 先将y_hat转为softmax形式，然后计算两者的损失
            l = tf.reduce_sum(loss)  # 将每个样本的loss求和
        grads = taps.gradient(l, [W, b])
        train_acc_num += accuracy(y_hat, y)
        train_l_sum += l
        n += X.shape[0]
        print("\r Epoch: {:d}/{:d} train loss: {:.4f} train acc: {:.4f} | {:.2%}".format(epoch, num_epochs,
                                                                                         train_l_sum / n,
                                                                                         train_acc_num / train_num,
                                                                                         n / train_num))
        sgd([W, b], lr, batch_size, grads)

    test_loss, test_acc = evaluate_accuracy(test_iter, model, W, b)
    print("\r Epoch: {:d}/{:d} train loss: {:.4f} train acc: {:.4f} | test loss: {:.4f} test acc: {:.4f}".format(epoch,
                                                                                                                 num_epochs,
                                                                                                                 train_l_sum / n,
                                                                                                                 train_acc_num / train_num,
                                                                                                                 test_loss,
                                                                                                                 test_acc))
