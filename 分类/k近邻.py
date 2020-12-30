# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:18:31 2020

@author: yx
"""
#获取数据
#从sklearn.datasets导入iris数据加载器
from sklearn.datasets import load_iris
#使用加载器读取数据并且存入变量iris。
iris = load_iris()
#查验数据规模
print(iris.data.shape)
#查看数据说明
print(iris.DESCR)

#数据分割
#从sklearn.model_selection里选择导入train_test_split用于数据分割
from sklearn.model_selection import train_test_split
#从使用train_test_split,利用随机种子random_state采样25%的数据作为测试集。
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.25,random_state=33)

#类别预测
#从sklearn.preprocessing里选择导入数据标准化模块
from sklearn.preprocessing import StandardScaler
#从sklearn.neighors里选择导入KNeigborsClassifier,即k近邻分类器。
from sklearn.neighbors import KNeighborsClassifier
#对训练课测试样本进行标准化
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
#使用k近邻分类器对测试数据进行类别预测，预测结果储存在变量y_predict中。
knc = KNeighborsClassifier()
knc.fit(x_train,y_train)
y_predict = knc.predict(x_test)

#性能评估
print(knc.score(x_test,y_test))
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict,target_names=iris.target_names))
