"""
@file:   xgboost模型
@author: FrankYi
@date:   2021/01/06
@desc:
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
titanic = pd.read_csv('train.csv')
#选取pclass、age以及sex作为训练特征
x = titanic[['Pclass', 'Age', 'Sex']]
y = titanic['Survived']

#对缺失的age信息，采用平均填方法进行补全，即以age列已知数据的平均数据填充。
x['Age'].fillna(x['Age'].mean(), inplace=True)

#分割数据
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

#特征向量化
vec = DictVectorizer(sparse=False)
x_train = vec.fit_transform(x_train.to_dict(orient='record'))
x_test = vec.transform(x_test.to_dict(orient='record'))

#使用随机森林训练
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
print(rfc.score(x_test, y_test))

#使用默认配置的xgboost模型进行预测
xgbc = XGBClassifier()
xgbc.fit(x_train, y_train)
print(xgbc.score(x_test, y_test))