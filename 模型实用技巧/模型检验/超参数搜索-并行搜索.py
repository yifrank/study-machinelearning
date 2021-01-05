"""
@file:   超参数搜索-并行搜索
@author: FrankYi
@date:   2021/01/05
@desc:
"""
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import  Pipeline
#导入网格搜索模块
from sklearn.model_selection import GridSearchCV
news = fetch_20newsgroups(subset='all')

x_train, x_test, y_train, y_test = train_test_split(news.data[:3000],news.target[:3000], test_size=0.25, random_state=33)

#使用pipeline简化系统搭建流程，将文本抽取与分类器模型串联起来
clf = Pipeline([('vect', TfidfVectorizer(stop_words='english', analyzer='word')),('svc',SVC())])
#这里需要试验的2个超参数的个数分别是4、3，svc_gamma的参数共有10^-2,10^-1……。这样我么一共有12种的超参数的组合，12个不同参数下的模型。
parameters = {'svc__gamma':np.logspace(-2, 1, 4),'svc__C':np.logspace(-1, 1,3)}

#将12组参数组合以及初始化的pipline包括3折交叉验证的要求全部告知Gridsearchcv.注意refit = True,n_jobs=-1代表使用该计算机全部的cpu。
gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3, n_jobs=-1)

#执行多线程并行网格搜索
gs.fit(x_train, y_train)
gs.best_params_,gs.best_score_

print(gs.score(x_test, y_test))