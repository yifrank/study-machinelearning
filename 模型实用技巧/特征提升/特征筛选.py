"""
@file:   特征筛选
@author: FrankYi
@date:   2020/12/31
@desc:
"""
import pandas as pd
titanic = pd.read_csv('train.csv')
#分离数据特征与预测目标
y = titanic['Survived']
x = titanic.drop(['Name', 'Survived'], axis=1)

#对缺失数据进行填充
x['Age'].fillna(x['Age'].mean(), inplace=True)
x.fillna('UNKNOWN', inplace=True)

#分割数据，依然采样25%用于测试
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

#类别型特征向量化
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
x_train = vec.fit_transform(x_train.to_dict(orient='record'))
x_test = vec.transform(x_test.to_dict(orient='record'))

#输出处理后特征向量的维度
print(len(vec.feature_names_))

#使用决策树模型依靠所有特征进行预测，并作性能评估
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
print(dt.score(x_test, y_test))

#导入特征选择器
from sklearn import feature_selection
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=40)
x_train_fs = fs.fit_transform(x_train ,y_train)
dt.fit(x_train_fs, y_train)
x_test_fs = fs.transform(x_test)
print(dt.score(x_test_fs, y_test))

from sklearn.model_selection import cross_val_score
import numpy as np
percentiles = range(1,100,2)
results = []
for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2,percentile=i)
    x_train_fs = fs.fit_transform(x_train, y_train)
    scores = cross_val_score(dt, x_train_fs, y_train, cv=5)
    results = np.append(results, scores.mean())
print(results)

#寻找性能最佳特征筛选百分比
opt = np.where(results == results.max())[0]
print('this is opt',opt[0])

import pylab as pl
pl.plot(percentiles, results)
pl.xlabel('percentiles of features')
pl.ylabel('accuracy')
pl.show()

#使用最佳筛选后的特征，利用相同配置模型在测试集上进行性能评估
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=percentiles[opt[0]])
x_train_fs = fs.fit_transform(x_train, y_train)
dt.fit(x_train_fs, y_train)
x_test_fs = fs.transform(x_test)
print(dt.score(x_test_fs, y_test))