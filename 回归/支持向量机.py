"""
@file:   支持向量机
@author: FrankYi
@date:   2020/12/26
@desc:
"""
#导入数据
from sklearn.datasets import load_boston
#从读取房价数据存储在变量boston中。
boston = load_boston()
#查看数据
print(boston.DESCR)

#数据分割
from sklearn.model_selection import train_test_split
import numpy as np
x = boston.data
y = boston.target
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=33)

#导入支持向量机（回归）模型
from sklearn.svm import SVR
#使用不同的核函数进行
#线性核函数
# linear_svr = SVR(kernel='linear')
# linear_svr.fit(x_train,y_train)
# linear_svr_y_predict = linear_svr.predict(x_test)

#多项式核函数
# poly_svr = SVR(kernel='')
# poly_svr.fit(x_train,y_train)
# poly_svr_y_predict = poly_svr.predict(x_test)

#径向基核函数
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(x_train,y_train)
rbf_svr_y_predict = rbf_svr.predict(x_test)

#性能评估
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
# print(linear_svr.score(x_test,y_test))
# print(r2_score(y_test,linear_svr_y_predict))
# print(mean_squared_error(y_test,linear_svr_y_predict))
# print(mean_absolute_error(y_test,linear_svr_y_predict))

# print(poly_svr.score(x_test,y_test))
# print(r2_score(y_test,poly_svr_y_predict))
# print(mean_squared_error(y_test,poly_svr_y_predict))
# print(mean_absolute_error(y_test,poly_svr_y_predict))

print(rbf_svr.score(x_test,y_test))
print(r2_score(y_test,rbf_svr_y_predict))
print(mean_squared_error(y_test,rbf_svr_y_predict))
print(mean_absolute_error(y_test,rbf_svr_y_predict))


"""
特点分析：
该系列模型还是可以通过配置不同的核函数来改变模型性能。因此，
我们在使用该类模型时，可以尝试不同的配置。
"""