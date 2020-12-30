"""
@file:   test
@author: FrankYi
@date:   2020/12/30
@desc:
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#read dataset
digits_train = pd.read_csv('optdigits.tra', header=None)
digits_test = pd.read_csv('optdigits.tes', header=None)

x_train = digits_train[np.arange(64)]
y_train = digits_train[64]
x_test = digits_test[np.arange(64)]
y_test = digits_test[64]

import time
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

svc = LinearSVC()
start = time.time()
svc.fit(x_train, y_train)
end = time.time()
y_predict = svc.predict(x_test)

estimator = PCA(n_components=20)
pca_x_train = estimator.fit_transform(x_train)
pca_x_test = estimator.transform(x_test)

pca_svc = LinearSVC()
start1 = time.time()
pca_svc.fit(pca_x_train,y_train)
end1 = time.time()
pca_y_predict = pca_svc.predict(pca_x_test)


print('raw data to train costs time:',end-start)
print('pca and to train costs time:',end1-start1)


