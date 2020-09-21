import tensorflow as tf

# 查看版本
print(tf.__version__)


#2.2 数据操作
## 2.2.1 创建tensor

x = tf.constant(range(12))
print(x)
print(type(x))
print(x.shape)
print(x.get_shape())
print(x.dtype)
print(x.device)
# print(x.graph)
# print(x.name)

## reshape
X = tf.reshape(x, (3, 4))
print(X.shape)
# X = x.reshape((4, -1))  EagerTensor类型 不支持
# print(X.shape)
## 创建零
zero_tensor = tf.zeros((2, 3, 4))
print(zero_tensor)
one_tensor = tf.ones((2, 3, 4))
print(one_tensor)

## 传入list创建tensor
Y = tf.constant([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
print(Y)
Y = tf.dtypes.cast(Y, tf.float32)
print(Y)

## 创建随机数据
Z = tf.random.normal(shape=[3,4], mean=0, stddev=1)
print(Z)

## 2.2.2 运算符
## 普通操作
X = tf.dtypes.cast(X, tf.float32)
result = X + Y
result = X * Y
result = X / Y
result = tf.exp(Y)
result = tf.matmul(X, tf.transpose(Y))

## 多个tensor连接concatenate
result = tf.concat([X, Y], axis=0)
print(result)

## 判断相同
print(X == Y)
result = tf.equal(X, Y)
print(result)

## 所有元素求和
eq_num = tf.reduce_sum(tf.cast(result, tf.int32))
print(eq_num)
eq_num = tf.norm(tf.cast(result, tf.float32))
print(eq_num)

## 2.2.3 广播机制
A = tf.reshape(tf.constant(range(3)), (3,1))
B = tf.reshape(tf.constant(range(2)), (1,2))
add_result = A + B
print(add_result)

## 2.2.4 索引
print(X)
print(X[1:])
## 索引赋值
X = tf.Variable(X)
X[1,2].assign(9.0)
## 多元素索引
print(tf.gather(X, [0,2]))

## 多索引赋值
X[1:2,:].assign(tf.ones(X[1:2,:].shape, dtype = tf.float32)*12)
print(X)

## 2.2.5 运算的内存开销
X = tf.Variable(X)
Y = tf.cast(Y, dtype=tf.float32)

before = id(Y)
Y = Y + X  #Y + X相加之后会重新开辟内存然后将Y指向新内存
print(id(Y) == before)

Z = tf.Variable(tf.zeros_like(Y))
before = id(Z)
Z[:].assign(X+Y) #X+Y相加时开了临时内存来存储计算结果，然后再复制给Z对应的内存中
print(id(Z) == before)
## 
Z = tf.add(X, Y)
print(id(Z) == before)

## 减少内存开支方法
before = id(X)
X[:].assign(X + Y)
print(id(X) == before)

# X += Y 错误用法
# print(id(X) == before)

X.assign_add(Y)
print(id(X) == before)

## 2.2.6 tensor和Numpy相互变换
import numpy as np
P = np.ones((2, 3))
D = tf.constant(P) # numpy 转tensor
print(D)

P = np.array(D) # tensor 转 numpy
print(P)

#2.3 自动求梯度
x = tf.reshape(tf.Variable(range(4), dtype=tf.float32),(4,1))
with tf.GradientTape() as t:
    #GradientTape默认只监控由tf.Variable创建的traiable=True属性（默认）的变量
    #如果需要监视常量的梯度，则需要将其加入到watch中
    t.watch(x) # watch函数把需要计算梯度的变量x加进来了
    y = 2 * tf.matmul(tf.transpose(x), x)

dy_dx = t.gradient(y, x)
print(dy_dx)

## 默认情况下GradientTape的资源在调用gradient函数后就被释放，再次调用就无法计算了，
# 如果需要计算多次梯度，需要开启persistent=True属性
x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as g:
  g.watch(x)
  y = x * x
  z = y * y
dz_dx = g.gradient(z, x)  # z = y^2 = x^4, z’ = 4*x^3 = 4*3^3
dy_dx = g.gradient(y, x)  # y’ = 2*x = 2*3 = 6
print(dy_dx)
print(dz_dx)

## 进行多阶导数计算
x = tf.constant(3.0)
with tf.GradientTape() as g:
  g.watch(x)
  with tf.GradientTape() as gg:
    gg.watch(x)
    y = x * x
  dy_dx = gg.gradient(y, x)      # y’ = 2*x = 2*3 =6
d2y_dx2 = g.gradient(dy_dx, x)  # y’’ = 2

## 最后，一般在网络中使用时，不需要显式调用watch函数，使用默认设置，
# GradientTape会监控可训练变量，例如：
with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
gradients = tape.gradient(loss, model.trainable_variables)

## 2.3.3 对python控制流求梯度
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c

x = tf.random.normal((1,1),dtype=tf.float32)
with tf.GradientTape() as t:
    t.watch(x)
    c = f(x)  # c = f(x) = k * x
t.gradient(c,x) == c/x


## 2.4 查阅文档
## 查看一个模块里面提供哪些可以调用的函数和类的时候，可以使用dir函数
## 想了解某个函数或类的具体用法时，可以使用help函数