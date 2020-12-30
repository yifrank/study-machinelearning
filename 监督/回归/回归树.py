"""
@file:   回归树
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

#导入模型
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(x_train,y_train)
dtr_y_predict = dtr.predict(x_test)

#性能评估
from sklearn.metrics import mean_squared_error, mean_absolute_error
print(dtr.score(x_test,y_test))
print(mean_absolute_error(y_test,dtr_y_predict))
print(mean_squared_error(y_test,dtr_y_predict))

"""
特点分析：
树模型的优点：
1、数模型可以解决非线性特征的问题
2、树模型不要求对特征标准化和统一量化，即数值型和类别型特征都可以直接被应用在树模型的构建和预测过程中
3、因为上述原因，树模型也可以直观地输出决策过程，使得预测结果具有可解释性

缺点：
1、正是因为树模型可以解决复杂的非线性拟合问题，所以更加容易因为模型搭建过于复杂而丧失对新数据预测的精度（泛化力）
2、树模型从上至下的预测流程会因为数据细微的更改而发生较大的结构变化，因此预测稳定性较差
3、依托训练数据构建最佳的数模型是NP难问题，即在有限时间内无法找到最优解的问题，因此我们所使用类似贪婪算法的解法只能找到一些次优解，这也是为什么我们经常借助集成模型，在多个次优解中寻觅更高的模型性能  
"""

