"""
@file:   词向量技术
@author: FrankYi
@date:   2021/01/05
@desc:
"""
from sklearn.datasets import fetch_20newsgroups
from bs4 import BeautifulSoup
import nltk, re
from gensim.models import word2vec
news = fetch_20newsgroups(subset='all')
x,y = news.data, news.target

#定义函数，将每条新闻中的局子注意剥离出来，并返回一个句子的列表。
def news_to_sentences(news):
    news_text = BeautifulSoup(news).get_text()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(news_text)
    sentences = []
    for sent in raw_sentences:
        sentences.append(re.sub('[^a-zA-Z]', ' ',sent.lower().strip()).split())
    return sentences
sentences = []

#将长篇新闻文本中的句子剥离出来
for i in x:
    sentences += news_to_sentences(i)
#配置词向量的维度
num_features = 300
#保证被考虑的词汇的频度
min_word_count = 20
#设定并行化训练使用cpu计算核心的数量，多核使用。
num_workers = 2
#定义训练词向量的上下文窗口大小
context = 5
downsampling = 1e-3

#训练词向量模型
model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, window=context, sample=downsampling)

model.init_sims(replace=True)

#利用训练好的模型，寻找训练文本中与morning最相关的10个词汇
print(model.most_similar('morning'))