#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
测试文件

@Author: xiezhenwei
@Date: 2017/12/6
"""

import bayes

listOPosts,listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)
print myVocabList
print bayes.setOfWords2Vec(myVocabList,listOPosts[0])