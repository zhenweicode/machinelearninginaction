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

bayes.spamTest()