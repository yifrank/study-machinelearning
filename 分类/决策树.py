"""
@file:   决策树
@author: FrankYi
@date:   2020/12/24
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

#从sklearn.tree中导入决策树分类器
from sklearn.tree import DecisionTreeClassifier
#使用默认配置初始化决策树分类器
dtc = DecisionTreeClassifier()
#训练模型
dtc.fit(x_train,y_train)
#模型预测
y_predict = dtc.predict(x_test)

#性能测评
from sklearn.metrics import classification_report
print(dtc.score(x_test,y_test))
print(classification_report(y_predict,y_test,target_names=['died','survived']))
