"""
@file:   tensorflow框架
@author: FrankYi
@date:   2021/01/06
@desc:
"""
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()
#初始化一个常量
greeting = tf.constant('Hello Google Tensorflow!')
#启动一个会话
sess = tf.Session()
#使用会话执行greeting计算模块
result = sess.run(greeting)
print(result)
sess.close()

#使用TensorFlow完成一次线性函数的计算
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
#product将上述两个算子相乘，作为新算例
product = tf.matmul(matrix1, matrix2)
#继续将product与一个标量2.0求和拼接，作为最终的linear算例、
linear = tf.add(product, tf.constant(2.0))

#直接在会话中执行linear算例，相当于将上面所有的单独算例拼接成流程图来执行。
with tf.Session() as sess:
    result = sess.run(linear)
    print(result)

#使用TensorFlow自定义一个线性分类器
import pandas as pd
from sklearn.model_selection import train_test_split
column_names = ['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
data = pd.read_csv('breast-cancer-wisconsin.data',names = column_names)
x_train,x_test,y_train,y_test = train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)
x_train = np.float32(x_train[['Clump Thickness', 'Uniformity of Cell Size']].T)
y_train = np.float32(y_train.T)
x_test = np.float32(x_test[['Clump Thickness', 'Uniformity of Cell Size']].T)
y_test = np.float32(y_test.T)
#定义一个TensorFlow的变量b作为线性模型的截距，同时设置为1.0
b = tf.Variable(tf.zeros([1]))
#定义一个TensorFlow的变量w作为线性模型的系数，并设置初始值为-1.0到1.0之间均匀分布的随机数
W = tf.Variable(tf.random_uniform([1,2], -1.0, 1.0))
#显示定义这个线性函数
y = tf.matmul(W, x_train) + b

loss = tf.reduce_mean(tf.square(y - y_train))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
#迭代1000轮次，训练参数
for step in range(0,1000):
    sess.run(train)
    if step % 200 == 0:
        print(step, sess.run(W), sess.run(b))

# import matplotlib.pyplot as plt
# y_test = pd.DataFrame(y_test, columns=['Type'])
# print(y_test, len(x_test.T))
# print(x_test[y_test.loc[y_test.Type == 2].index,:], y_test)
# plt.scatter(x_test[0,y_test],x_test[1,:],marker='o',s=200, c='red')
# plt.xlabel('Clump Thickness')
# plt.ylabel('Cell Size')
# plt.show()