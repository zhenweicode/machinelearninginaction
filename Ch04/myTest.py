#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
测试文件

@Author: xiezhenwei
@Date: 2017/12/6
"""

import bayes

# listOPosts, listClasses = bayes.loadDataSet()
# myVocabList = bayes.createVocabList(listOPosts)
# print myVocabList
# print bayes.setOfWords2Vec(myVocabList, listOPosts[0])
#
# trainMat = []
# for postinDoc in listOPosts:  # 使用词向量来填充trainMat列表
#     trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))
#
# # 属于侮辱性文档的概率以及两个类别的概率向量
# p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)
# print p0V
# print p1V
# print pAb
#
# bayes.testingNB()

# bayes.spamTest()



# import feedparser
#
# rss_url = 'http://www.oschina.net/news/rss'
#
# # 获得订阅
# feeds = feedparser.parse(rss_url)
#
# # 获得rss版本
# print(feeds.version)
# # 获得Http头
# print(feeds.headers)
# print(feeds.headers.get('content-type'))
#
# # rss的标题
# print(feeds['feed']['title'])
# # 链接
# print(feeds['feed']['link'])
# # 子标题
# print(feeds.feed.subtitle)
# # 查看文章数量
# print(len(feeds['entries']))
# # 获得第一篇文章的标题
# print(feeds['entries'][0]['title'])
# # 获得第一篇文章的链接
# print(feeds.entries[0]['link'])
