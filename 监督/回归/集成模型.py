"""
@file:   集成模型
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

#导入模型和训练
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)
rfr_y_predict = rfr.predict(x_test)

etr = ExtraTreesRegressor()
etr.fit(x_train, y_train)
etr_y_predict = etr.predict(x_test)

gbr = GradientBoostingRegressor()
gbr.fit(x_train, y_train)
gbr_y_predict = gbr.predict(x_test)

#性能评估
from sklearn.metrics import mean_absolute_error, mean_squared_error
print(rfr.score(x_test, y_test))
print(mean_absolute_error(y_test, rfr_y_predict))
print(mean_squared_error(y_test, rfr_y_predict))

print(etr.score(x_test, y_test))
print(mean_absolute_error(y_test, etr_y_predict))
print(mean_squared_error(y_test, etr_y_predict))

print(gbr.score(x_test, y_test))
print(mean_absolute_error(y_test, gbr_y_predict))
print(mean_squared_error(y_test, gbr_y_predict))
