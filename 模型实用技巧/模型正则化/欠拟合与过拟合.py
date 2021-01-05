"""
@file:   欠拟合与过拟合
@author: FrankYi
@date:   2021/01/03
@desc:
"""
x_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]
#进行线性回归
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#训练数据
regressor.fit(x_train, y_train)

import numpy as np
xx = np.linspace(0,26,100)
xx = xx.reshape(xx.shape[0], 1)
yy = regressor.predict(xx)

import matplotlib.pyplot as plt
plt.scatter(x_train, y_train)
plt1, = plt.plot(xx, yy, label='Degree=1')
plt.axis([0,25,0,25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt1])
plt.show()

print(regressor.score(x_train, y_train))

from sklearn.preprocessing import PolynomialFeatures
poly2 = PolynomialFeatures(degree=2)
x_train_poly2 = poly2.fit_transform(x_train)
regressor_poly2 = LinearRegression()
regressor_poly2.fit(x_train_poly2, y_train)
xx_poly2 = poly2.transform(xx)
yy_poly2 = regressor_poly2.predict(xx_poly2)

#绘图
plt.scatter(x_train, y_train)
plt1, = plt.plot(xx, yy, label='Degree=1')
plt2, = plt.plot(xx, yy_poly2, label='Degree=2')

plt.axis([0,25,0,25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt1, plt2])
plt.show()

print(regressor_poly2.score(x_train_poly2, y_train))


poly4 = PolynomialFeatures(degree=4)
x_train_poly4 = poly4.fit_transform(x_train)
regressor_poly4 = LinearRegression()
regressor_poly4.fit(x_train_poly4, y_train)
xx_poly4 = poly4.transform(xx)
yy_poly4 = regressor_poly4.predict(xx_poly4)

#绘图
plt.scatter(x_train, y_train)
plt1, = plt.plot(xx, yy, label='Degree=1')
plt2, = plt.plot(xx, yy_poly2, label='Degree=2')
plt4, = plt.plot(xx, yy_poly4, label='Degree=4')

plt.axis([0,25,0,25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt1, plt2, plt4])
plt.show()

print(regressor_poly4.score(x_train_poly4, y_train))

x_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

print(regressor.score(x_test, y_test))

x_test_poly2 = poly2.transform(x_test)
print(regressor_poly2.score(x_test_poly2, y_test))

x_test_poly4 = poly4.transform(x_test)
print(regressor_poly4.score(x_test_poly4, y_test))

print(regressor_poly4.coef_)