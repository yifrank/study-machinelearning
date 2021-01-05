"""
@file:   L1范数正则化
@author: FrankYi
@date:   2021/01/04
@desc:
"""
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
x_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]
x_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

poly4 = PolynomialFeatures(degree=4)
x_train_poly4 = poly4.fit_transform(x_train)
lasso_poly4 = Lasso()
lasso_poly4.fit(x_train_poly4, y_train)
x_test_poly4 = poly4.transform(x_test)
print(lasso_poly4.score(x_test_poly4, y_test))

print(lasso_poly4.coef_)
