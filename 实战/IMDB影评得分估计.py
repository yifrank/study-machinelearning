"""
@file:   IMDB影评得分估计
@author: FrankYi
@date:   2021/01/09
@desc:
"""
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import nltk.data
from gensim.models import word2vec
from gensim.models import Word2Vec
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
train = pd.read_csv('./IMDB/labeledTrainData.tsv', delimiter='\t')
test = pd.read_csv('./IMDB/testData.tsv', delimiter='\t')

print(train.head())
print(test.head())

def review_to_text(review, remove_stopwords):
#任务1：去掉html标记
    raw_text = BeautifulSoup(review, 'html').get_text()
#任务2：去掉非字母字符
    letters = re.sub('[^a-zA-Z]', ' ', raw_text)
    words = letters.lower().split()
#任务3：如果remove_stopwords被激活，则进一步去掉评论中的停用词
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
#返回每条评论经此三项预处理的词汇列表
    return words

#分别对原始训练和数据集进行上述三项预处理
x_train = []
for review in train['review']:
    x_train.append(' '.join(review_to_text(review, True)))
x_test = []
for review in test['review']:
    x_test.append(' '.join(review_to_text(review, True)))

y_train = train['sentiment']

pip_count = Pipeline([('count_vec', CountVectorizer(analyzer='word')), ('mnb', MultinomialNB())])
pip_tfidf = Pipeline([('tfidf_vec', TfidfVectorizer(analyzer='word')), ('mnb', MultinomialNB())])

params_count = {'count_vec__binary': [True, False], 'count_vec__ngram_range': [(1,1),(1,2)], 'mnb__alpha':[0.1, 1.0, 10.0]}
params_tfidef = {'tfidf_vec__binary': [True, False], 'tfidf_vec__ngram_range': [(1,1),(1,2)], 'mnb__alpha':[0.1, 1.0, 10.0]}

#4折交叉验证进行超参数搜索
gs_count = GridSearchCV(pip_count, params_count, cv=4, n_jobs=-1, verbose=1)
gs_count.fit(x_train, y_train)

print(gs_count.best_score_)
print(gs_count.best_params_)

count_y_predict = gs_count.predict(x_test)

gs_tfidf = GridSearchCV(pip_tfidf, params_tfidef, cv=4, n_jobs=-1, verbose=1)
gs_tfidf.fit(x_train, y_train)

print(gs_tfidf.best_params_)
print(gs_tfidf.best_score_)

tfidf_y_predict = gs_tfidf.predict(x_test)

#使用pandas对需要提交的数据进行格式化
submission_count = pd.DataFrame({'id': test['id'], 'sentiment': count_y_predict})
submission_tfidf = pd.DataFrame({'id': test['id'], 'sentiment': tfidf_y_predict})

#保存数据到本地
submission_count.to_csv('./IMDB/submission_count.csv', index=False)
submission_tfidf.to_csv('./IMDB/submission_tfidf.csv', index=False)

#从本地读入未标记的数据
unlabeled_train = pd.read_csv('./IMDB/unlabeledTrainData.tsv', delimiter='\t', quoting=3)
#准备使用nltk的tokenizer对影评中的英文句子进行分割
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#定义函数review_to_sentences逐条对影评进行分句
def review_to_sentences(review, tokenizer):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_text(raw_sentence, False))
    return sentences

corpora = []
#准备用于训练词向量的数据
for review in unlabeled_train['review']:
    corpora += review_to_sentences(review, tokenizer)

#配置训练词向量的超参数
num_features = 300
min_word_count = 20
num_workers = 4
context = 10
downsampling = 1e-3

model = word2vec.Word2Vec(corpora, workers=num_workers, size=num_features, min_count=min_word_count, window=context, sample=downsampling)

model.init_sims(replace=True)
model_name = "./IMDB/300features_20minwords_10context"
model.save(model_name)

model = Word2Vec.load("./IMDB/300features_20minwords_10context")

#探查一下该词向量模型的训练成果
model.most_similar('man')

#定义一个函数词向量产生文本特征向量
def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype='float32')
    nwords = 0
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec, model[word])
    featureVec = np.divide(featureVec, nwords)
    return featureVec

#定义一个每条影评转化为基于词向量的特征向量（平均词向量）
def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype='float32')
    for review in reviews:
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter += 1
    return reviewFeatureVecs

#准备新的基于词向量表示的训练和测试特征向量
clean_train_reviews = []
for review in train['review']:
    clean_train_reviews.append(review_to_text(review, remove_stopwords=True))
trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

clean_test_reviews = []
for review in test['review']:
    clean_test_reviews.append(review_to_text(review, remove_stopwords=True))
testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)

#使用GradientBoostingClassifierz模型
gbc = GradientBoostingClassifier()
params_gbc = {'n_estimators':[10, 100, 500], 'learning_rate':[0.01, 0.1,1.0], 'max_depth': [2,3,4]}
gs = GridSearchCV(gbc, params_gbc, cv=4, n_jobs=-1, verbose=1)

gs.fit(trainDataVecs, y_train)
print(gs.best_score_)
print(gs.best_params_)

result = gs.predict(testDataVecs)
output = pd.DataFrame(data={'id':test['id'],'sentiment': result})
output.to_csv('./IMDB/submission_w2v.csv', index=False, quoting=3)
