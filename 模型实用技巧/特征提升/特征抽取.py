"""
@file:   特征抽取
@author: FrankYi
@date:   2020/12/31
@desc:
"""
#使用DictVectorizer对字典存储的数据进行特征抽取与向量化
#定义一组字典列表，用来表示多个数据样本（每个字典代表一个数据样本）
measurements = [{'city':'Dubai','temperature': 33.},{'city':'London','temperature': 12.},{'city':'San Fransisco','temperature': 18.}]
#导入DictVectorizer
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
print(vec.fit_transform(measurements).toarray())
print(vec.get_feature_names())

#导入数据
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all')
#分割数据
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer()
#只使用词频统计的方式将原始训练和测试文本转化为特征向量
x_count_train = count_vec.fit_transform(x_train)
x_count_test = count_vec.transform(x_test)

#导入模型
from sklearn.naive_bayes import MultinomialNB
mnb_count = MultinomialNB()
mnb_count.fit(x_count_train, y_train)
print(mnb_count.score(x_count_test, y_test))
y_count_predict = mnb_count.predict(x_count_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_count_predict, target_names=news.target_names))

#TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer()
x_tfidf_train = tfidf_vec.fit_transform(x_train)
x_tfidf_test = tfidf_vec.transform(x_test)

mnb_tfidf = MultinomialNB()
mnb_tfidf.fit(x_tfidf_train, y_train)
print(mnb_tfidf.score(x_tfidf_test, y_test))
y_tfidf_predict = mnb_tfidf.predict(x_tfidf_test)
print(classification_report(y_test, y_tfidf_predict, target_names=news.target_names))

#过滤停用词
count_filter_vec, tfidf_filter_vec = CountVectorizer(analyzer='word', stop_words='english'), TfidfVectorizer(analyzer='word', stop_words='english')

x_count_filter_train = count_filter_vec.fit_transform(x_train)
x_count_filter_test = count_filter_vec.transform(x_test)

x_tfidf_filter_train = tfidf_filter_vec.fit_transform(x_train)
x_tfidf_filter_test = tfidf_filter_vec.transform(x_test)

#训练
mnb_count_filter = MultinomialNB()
mnb_count_filter.fit(x_count_filter_train, y_train)
print(mnb_count_filter.score(x_count_filter_test, y_test))
y_count_filter_predict = mnb_count_filter.predict(x_count_filter_test)

mnb_tfidf_filter = MultinomialNB()
mnb_tfidf_filter.fit(x_tfidf_filter_train, y_train)
print(mnb_tfidf_filter.score(x_tfidf_filter_test, y_test))
y_tfidf_filter_predict = mnb_tfidf_filter.predict(x_tfidf_filter_test)

#性能评估
print(classification_report(y_test, y_count_filter_predict, target_names=news.target_names))
print(classification_report(y_test, y_tfidf_filter_predict, target_names=news.target_names))
