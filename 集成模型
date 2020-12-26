"""
@file:   集成模型
@author: FrankYi
@date:   2020/12/25
@desc:
"""
#导入pandas用于数据分析
import pandas as pd
#利用pandas的read_csv模块直接从互联网收集泰坦尼克号乘客的信息。
titanic = pd.read_csv('train.csv')
#观察数据
print(titanic.head())
print(titanic.info())

#预测生还情况
#特征的选择，机器学习中重要的一步工作
x = titanic[['Pclass', 'Age', 'Sex']]
y = titanic['Survived']
#对当前选择的特征进行探查
print(x.info())

#借由上面的输出，我们设计如下几个数据处理任务
#1) age这个数据列只有714个，需要补充完。
#2) sex数据列的值，需要转化为数值特征，用0/1代替。

#首先来不从age里的数据，使用平均数或者中位数都是对模型偏离造成最小影响的策略。
x['Age'].fillna(x['Age'].mean(), inplace=True)
#查看补充完的数据
print(x.info())

#数据分割
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=33)

#使用sklea.feature_extraction中的特征转化器。
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
#转换特征后，我们发现凡是类别型的特征都单独剥离出来，独成一列特征，数值型的不变。
x_train = vec.fit_transform(x_train.to_dict(orient='record'))
print(vec.feature_names_)

#同样需要对测试数据的特征进行转换。
x_test = vec.transform(x_test.to_dict(orient='record'))

#使用单一决策树
#从sklearn.tree中导入决策树分类器
from sklearn.tree import DecisionTreeClassifier
#使用默认配置初始化决策树分类器
dtc = DecisionTreeClassifier()
#训练模型
dtc.fit(x_train,y_train)
#模型预测
dtc_y_predict = dtc.predict(x_test)

#使用随机森林分类器
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
rfc_y_predict = rfc.predict(x_test)

#使用梯度提升决策树
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(x_train,y_train)
gbc_y_predict = gbc.predict(x_test)
#性能测评
from sklearn.metrics import classification_report
print(dtc.score(x_test,y_test))
print(rfc.score(x_test,y_test))
print(gbc.score(x_test,y_test))
print(classification_report(dtc_y_predict,y_test,target_names=['died','survived']))
print(classification_report(rfc_y_predict,y_test,target_names=['died','survived']))
print(classification_report(gbc_y_predict,y_test,target_names=['died','survived']))


"""特点分析：
集成模型可以说是实战应用中最为常见的。相比于其他单一的学习模型，集成模型可以整合多种
模型，或者多次就一种类型的模型进行建模。由于模型估计参数的过程也同样受到概率的影响，具有一定的
不确定性；因此，集成模型虽然在训练过程中需要耗费更多的时间，但是得到的综合模型往往具有更高的表现性和更好的稳定性。
"""
