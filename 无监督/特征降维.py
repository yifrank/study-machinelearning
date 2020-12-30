"""
@file:   特征降维-PCA
@author: FrankYi
@date:   2020/12/30
@desc:
"""
#前言
#线性相关矩阵秩计算
import numpy as np
#初始化一个2*2的线性相关矩阵
M = np.array([[1,2],[2,4]])
#计算2*2线性相关的秩
print(np.linalg.matrix_rank(M, tol=None))

#显示手写体数字图片经PCA压缩后的二维空间分布
#分别导入numpy、matplotlib以及pandas,用于数学运算、作图以及数据分析。
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#read dataset
digits_train = pd.read_csv('optdigits.tra', header=None)
digits_test = pd.read_csv('optdigits.tes', header=None)
#分割训练数据的特征向量和标记
x_digits = digits_train[np.arange(64)]
y_digits = digits_train[64]
#导入PCA
from sklearn.decomposition import PCA
estimator = PCA(n_components=2)
X_pca = estimator.fit_transform(x_digits)

#显示10类手写体数字图片经pca压缩后的2位空间分布
def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow','white','red','lime','cyan','orange','gray']
    for i in range(len(colors)):
        px = X_pca[:, 0][y_digits.values == i]
        py = X_pca[:, 1][y_digits.values == i]
        plt.scatter(px,py,c=colors[i])
    plt.legend(np.arange(0,10).astype(str))
    plt.xlabel('First Prinipal Component')
    plt.ylabel('Second Principal Component')
    plt.show()

plot_pca_scatter()

#使用PCA降维，对比svm分类的差异
x_train = digits_train[np.arange(64)]
y_train = digits_train[64]
x_test = digits_test[np.arange(64)]
y_test = digits_test[64]
#导入svm分类器
import time
from sklearn.svm import LinearSVC
svc = LinearSVC()
start = time.time()
svc.fit(x_train, y_train)
end = time.time()
y_predict = svc.predict(x_test)

estimator = PCA(n_components=20)
pca_x_train = estimator.fit_transform(x_train)
pca_x_test = estimator.transform(x_test)
#pca之后的模型训练
pca_svc = LinearSVC()
start1 = time.time()
pca_svc.fit(pca_x_train,y_train)
end1 = time.time()
pca_y_predict = pca_svc.predict(pca_x_test)


#性能评估
from sklearn.metrics import classification_report
a = end-start
print('raw data to train costs time:',a)
print(classification_report(y_test, y_predict, target_names=list(np.arange(10).astype(str))))
#pca
b = end1 - start1
print('pca and to train costs time:', b)
print(classification_report(y_test, pca_y_predict, target_names=list(np.arange(10).astype(str))))

"""
特点分析：
降维/压缩问题则是选取数据具有代表性的特征，在保持数据多样性的基础上，
规避掉大量的特征冗余和噪声，不过这个过程也是很有可能会损失一些有用的模式信息。
经过大量的实践证明，相较于损失的少部分模型性能，维度压缩能够节省大量用于模型训练的时间。这样
一来，使得pca所带来的的模型综合效率变得更为划算。
"""