#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
测试

@Author: xiezhenwei
@Date: 2017/12/10
"""

import trees

dataSet, labels = trees.createDataSet()
# print trees.calcShannonEnt(dataSet)
# print trees.splitDataSet(dataSet, 0, 1)

print trees.createTree(dataSet, labels)