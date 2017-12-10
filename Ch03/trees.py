#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
决策树算法

@Author: xie
@Date: 2017/12/10
"""
from math import log
import operator

"""
简单的数据集，注意dataSet中前两个表示特征，最后一个是划分结果。现在我们想要决定首先依据第一个特征还是第二个特征划分数据
"""
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']   #注意labels的含义和第二章不太一样，第二章表示样本点的分类，这里表示特征的个数
    return dataSet, labels

"""
计算给定数据集的香农熵
"""
def calcShannonEnt(dataSet):
    #为所有可能分类创建字典
    numEntries = len(dataSet)   #计算数据集中实例的总数
    labelCounts = {}   #创建数据字典，它的键值是最后一列的数值，即最后一列表示分类
    for featVec in dataSet:
        currentLabel = featVec[-1]
        """
        如果当前键值不存在，则扩展字典并将当前键值加入字典。
        每个键值都记录了当前类别出现的次数。最后，使用所有类标签的发生频率计算类别出现的概率。
        我们将用这个概率计算香农熵，统计所有类标签发生的次数。
        """
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    #计算信息熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log以2为底
    return shannonEnt

"""
按照给定特征划分数据集
三个输入：待划分的数据集、划分数据集的特征（作为特征的某一列）、需要返回的特征的值（这一列的某一个特征值，等于这个特征值的行会被返回）。
"""
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        #遍历数据集中的每个元素，一旦发现符合要求的值，则将其添加到新创建的列表中。
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

"""
选择最好的数据集划分方式,
该函数实现选取特征，划分数据集，计算得出最好的划分数据集的特征。
"""
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #特征的个数，最后一列表示分类标签，并不是特征
    baseEntropy = calcShannonEnt(dataSet) #计算整个数据集的原始香农熵，我们保存最初的无序度量值，用于与划分完之后的数据集计算的熵值进行比较
    bestInfoGain = 0.0; bestFeature = -1

    # 遍历数据集中的所有特征
    for i in range(numFeatures):
        # 使用列表推导(List Comprehension)来创建新的列表，将数据集中所有第i个特征值或者所有可能存在的值写入这个新list中
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)       #获取当前列表所有唯一属性值
        newEntropy = 0.0

        #遍历当前特征中的所有唯一属性值，对每个特征划分一次数据集，然后计算数据集的新熵值，并对所有唯一特征值得到的熵求和。
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))   #
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy   #信息增益是熵的减少或者是数据无序度的减少

        #计算最好的信息增益
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

"""
返回出现次数最多的分类名称
该函数输入分类名称的列表， 然后创建键值为classList中唯一值的数据字典，字典对象存储了classList中每个类标签出现的频率，
最后利用operator操作键值排序字典，并返回出现次数最多的分类名称
"""
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

"""
创建操作树的函数代码
两个输入参数:数据集和标签列表。标签列表包含了数据集中所有特征的标签，算法本身并不需要这个变量，但是为了给出数据明确的含义，我们将它作为一个输入参数提供
"""
def createTree(dataSet,labels):
    #遍历获得分类列表，即dataSet最后一列
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): #类别完全相同则停止继续划分
        return classList[0]
    # 表示遍历完所有特征，因为每一次遍历会将使用的特征列去除，所以长度为1，表示只剩分类列
    if len(dataSet[0]) == 1:
        #表示使用完所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。因此挑选出现次数最多的类别作为返回值。
        return majorityCnt(classList)
    #在剩下的特征列中，寻找获得最大信息增益，即最小信息熵的列
    bestFeat = chooseBestFeatureToSplit(dataSet)
    #最优特征列对应的特征名
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        #因为在Python语言中函数参数是列表类型时，参数是按照引用方式传递的。为了保证每次调用函数createTree()时不改变原始列表的内容，使用新变量subLabels代替原始列表。
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            
    
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
