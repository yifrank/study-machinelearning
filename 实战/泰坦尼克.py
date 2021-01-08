"""
@file:   泰坦尼克
@author: FrankYi
@date:   2021/01/07
@desc:
"""
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train.info())
print(test.info())
#选择特征
selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']
x_train = train[selected_features]
x_test = test[selected_features]

y_train = train['Survived']

print(x_train['Embarked'].value_counts())
print(x_test['Embarked'].value_counts())

#使用频率最高的特征值进行填充
x_train['Embarked'].fillna('S', inplace=True)
x_test['Embarked'].fillna('S', inplace=True)

#使用平均值进行填充age
x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)
x_test['Age'].fillna(x_test['Age'].mean(), inplace=True)
x_test['Fare'].fillna(x_test['Fare'].mean(), inplace=True)

print(x_train.info())
print(x_test.info())

dict_vec = DictVectorizer(sparse=False)

x_train = dict_vec.fit_transform(x_train.to_dict(orient='record'))
print(dict_vec.feature_names_)

x_test = dict_vec.transform(x_test.to_dict(orient='record'))

rfc = RandomForestClassifier()
xgbc = XGBClassifier()

#使用5折交叉验证的方法在训练集上分别对默认配置的RandomForestClassifier以及XGBClassifier进行性能评估，并获得平均分类准确性的得分。
print(cross_val_score(rfc, x_train, y_train, cv=5).mean())
print(cross_val_score(xgbc, x_train, y_train, cv=5).mean())

#使用默认配置的RandomRorestClassifier进行预测操作
rfc.fit(x_train, y_train)
rfc_y_predict = rfc.predict(x_test)
rfc_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': rfc_y_predict})
rfc_submission.to_csv('rfc_submission.csv', index=False)

#使用默认配置的XGBClassifier进行预测操作
xgbc.fit(x_train, y_train)
xgbc_y_predict = xgbc.predict(x_test)
xgbc_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc_y_predict})
xgbc_submission.to_csv('xgbc_submission.csv', index=False)

#并使用并行网格搜索的方式寻找更好的超参数组合，以期待进一步提高XGBlassifier的预测性能
params = {'max_depth': range(2,7),'n_estimators': range(100, 1100, 200), 'learning_rate':[0.05, 0.1, 0.25, 0.5, 1.0]}
xgbc_best = XGBClassifier()
gs = GridSearchCV(xgbc_best, params, n_jobs=1, cv=5, verbose=1)
gs.fit(x_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

xgbc_best_y_predict = gs.predict(x_test)
xgbc_best_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc_best_y_predict})
xgbc_best_submission.to_csv('xgbc_best_submission.csv', index=False)
