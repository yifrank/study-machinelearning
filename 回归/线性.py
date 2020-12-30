"""
@file:   线性
@author: FrankYi
@date:   2020/12/25
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
print("the max target value is ",np.max(y))
print("the min target value is ",np.min(y))
print("the average target value is ",np.mean(y))

#数据差异大，进行标准化
from sklearn.preprocessing import StandardScaler
#初始化标准器
ss_x = StandardScaler()
ss_y = StandardScaler()
#对数据的特征和
# x_train = ss_x.fit_transform(x_train)
# x_test = ss_x.transform(x_test)
# y_train = ss_y.fit_transform(y_train.reshape(1, -1))
# y_test = ss_y.transform(y_test.reshape(-1, 1))

#模型预测
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
#初始化模型
linear = LinearRegression()
sgd = SGDRegressor()
#训练模型
linear.fit(x_train,y_train)
sgd.fit(x_train,y_train)
#预测
l_y_predict = linear.predict(x_test)
s_y_predict = sgd.predict(x_test)

#性能评估
print(linear.score(x_test,y_test))
print(sgd.score(x_test,y_test))
#mae与mse和r-squared评价指标
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error
print(r2_score(y_test,l_y_predict))
print(mean_squared_error(y_test,l_y_predict))
print(mean_absolute_error(y_test,l_y_predict))

"""特点分析：
在不清楚特征之间关系的前提下，我们仍然可以使用可以使用线性回归模型作为大多数科学实验
的基线系统
"""