"""
@file:   L2范数正则化
@author: FrankYi
@date:   2021/01/04
@desc:
"""
import numpy as np
coef_poly4 = np.array([ 0.00000000e+00, -2.51739583e+01,  3.68906250e+00 ,-2.12760417e-01,4.29687500e-03])
print(coef_poly4)
print(np.sum(coef_poly4 ** 2))

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
x_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]
x_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

poly4 = PolynomialFeatures(degree=4)
x_train_poly4 = poly4.fit_transform(x_train)
x_test_poly4 = poly4.transform(x_test)

ridge_poly4 = Ridge()
ridge_poly4.fit(x_train_poly4, y_train)
print(ridge_poly4.score(x_test_poly4, y_test))
print(ridge_poly4.coef_)
print(np.sum(ridge_poly4.coef_ ** 2))
