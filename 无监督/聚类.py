"""
@file:   聚类-kmeans
@author: FrankYi
@date:   2020/12/30
@desc:
"""
#分别导入numpy、matplotlib以及pandas,用于数学运算、作图以及数据分析。
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#read dataset
digits_train = pd.read_csv('optdigits.tra', header=None)
digits_test = pd.read_csv('optdigits.tes', header=None)
#从训练与测试数据集上都分离出64维度的像素特征与1维度的数字目标。
x_train = digits_train[np.arange(64)]
y_train = digits_train[64]

x_test = digits_test[np.arange(64)]
y_test = digits_test[64]

#导入kmeans模型
from sklearn.cluster import KMeans
#init model
kmeans = KMeans(n_clusters=10)
kmeans.fit(x_train, y_train)
#predict
y_predict = kmeans.predict(x_test)

#性能评估
from sklearn import metrics
#use ARI
print(metrics.adjusted_rand_score(y_test, y_predict))

#使用轮廓系数评价
from sklearn.metrics import silhouette_score
#init raw data
plt.subplot(3,2,1)
x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
X = np.array(list(zip(x1, x2))).reshape(len(x1),2)

#在1号子图做出原始数据点阵的分布
plt.xlim([0,10])
plt.ylim([0,10])
plt.title('instances')
plt.scatter(x1,x2)

colors = ['b','g','r','c','m','y','k','b']
markers = ['o','s','D','v','^','p','*','+']

clusters = [2,3,4,5,8]
subplot_counter = 1
sc_scores = []
for t in clusters:
    subplot_counter += 1
    plt.subplot(3,2,subplot_counter)
    kmeans_model = KMeans(n_clusters=t).fit(X)
    for i,l in enumerate(kmeans_model.labels_):
        print(i,l)
        plt.plot(x1[i],x2[i],color = colors[l], marker = markers[l], ls='None')
    plt.xlim([0,10])
    plt.ylim([0,10])
    sc_score = silhouette_score(X, kmeans_model.labels_, metric='euclidean')
    sc_scores.append(sc_score)
    plt.title('K=%s, silhouette coefficient = %0.03f'%(t, sc_score))

plt.figure()
plt.plot(clusters, sc_scores, '*-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Cofficient Score')
plt.show()

"""
特点分析：
kmeans聚类模型所采用的迭代式算法，直观易懂并且非常实用。
存在两大缺陷：
1、容易收敛到局部最优解
2、需要预先设定簇的数量
解决方法
1、执行多次kmeans算法来挑选性能表现更好的初始化中心点
2、使用“肘部”观察法，粗略地预估相对合理的类簇个数。
"""
#“肘部”观察法
from scipy.spatial.distance import  cdist
#使用均匀分布函数随机三个簇，每个簇周围10个数据样本
cluster1 = np.random.uniform(0.5,1.5,(2,10))
cluster2 = np.random.uniform(5.5,6.5,(2,10))
cluster3 = np.random.uniform(3.0,4.0,(2,10))
#绘制30个数据样本的分布图像
Q = np.hstack((cluster1, cluster2, cluster3)).T
plt.scatter(Q[:,0],Q[:,1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

#测试9种不同聚类中心数量下，每种情况的聚类质量，并作图
K = range(1,10)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(Q)
    meandistortions.append(sum(np.min(cdist(Q,kmeans.cluster_centers_,'euclidean'), axis=1))/Q.shape[0])
plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average Dispersion')
plt.title('Selecting k with the Elbow Method')
plt.show()