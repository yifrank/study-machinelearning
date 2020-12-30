# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 16:11:57 2020

@author: yx
"""
#获取数据
#从sklearn.datasets里导入新闻数据抓取器fetch_20newsgroups.
from sklearn.datasets import fetch_20newsgroups
#与之前预存的数据不同，fetch_20newsgroups需要及时从互联网下载数据。
news = fetch_20newsgroups(subset='all')
#查验数据规模和细节
print(len(news.data))
print(news.data[0])

#数据分割
#从sklearn.model_selection导入train_test_split。
from sklearn.model_selection import train_test_split
#随机采样25%的数据样本作为测试集
x_train,x_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25,random_state=33)

#类别预测
#从sklearn.feature_extraction.text里导入用于文本特征向量转化模块
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
x_train = vec.fit_transform(x_train)
x_test = vec.transform(x_test)

#从sklearn.naive_bayes里导入朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
#从使用默认配置初始化朴素贝叶斯模型
mnb = MultinomialNB()
#利用训练数据对模型参数进行估计
mnb.fit(x_train,y_train)
#对测试样本进行类别预测，结果存储在变量y_predict中
y_predict = mnb.predict(x_test)

#性能评估
#从sklearn.metrics里导入classfication_report用于详细的分类性能报告
from sklearn.metrics import classification_report
print(mnb.score(x_test,y_test))
print(classification_report(y_test,y_predict,target_names=news.target_names))