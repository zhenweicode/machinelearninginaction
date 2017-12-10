#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
测试文件

@Author: xiezhenwei
@Date: 2017/12/10
"""
from numpy import array

import kNN

# group, labels = kNN.createDataSet()
# print kNN.classify0([0, 0], group, labels, 3)
#
# dataMat, labels = kNN.file2matrix("./datingTestSet2.txt")
# print dataMat
# print labels
#
# import matplotlib
# import matplotlib.pyplot as plt
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dataMat[:, 1], dataMat[:, 2], 15.0 * array(labels), 15.0 * array(labels))
# plt.show()

# print kNN.datingClassTest()

print kNN.handwritingClassTest()