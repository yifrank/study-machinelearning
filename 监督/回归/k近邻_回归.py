"""
@file:   k近邻_回归
@author: FrankYi
@date:   2020/12/27
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
x = boston.data
y = boston.target
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=33)

#导入KNeighborRegressor
from sklearn.neighbors import KNeighborsRegressor
#初始化模型
uni_knr = KNeighborsRegressor(weights='uniform')
uni_knr.fit(x_train,y_train)
uni_knr_y_predict = uni_knr.predict(x_test)

dis_knr = KNeighborsRegressor(weights='distance')
dis_knr.fit(x_train,y_train)
dis_knr_y_predict = uni_knr.predict(x_test)

#性能评估
from sklearn.metrics import mean_absolute_error,mean_squared_error
print(uni_knr.score(x_test,y_test))
print(mean_squared_error(y_test,uni_knr_y_predict))
print(mean_absolute_error(y_test,uni_knr_y_predict))
print(dis_knr.score(x_test,y_test))
print(mean_squared_error(y_test,dis_knr_y_predict))
print(mean_absolute_error(y_test,dis_knr_y_predict))

"""
特点分析：
k近邻（回归）与k近邻（分类）一样，均属于无参数模型，同样没有没有参数训练过程。
"""