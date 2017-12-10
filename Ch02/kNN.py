#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
k近邻算法

@Author: xie
@Date: 2017/12/5
"""

from numpy import *   #科学计算包
import operator       #运算符模块
from os import listdir #从os模块中导入函数listdir，它可以列出给定目录的文件名

"""
作用：使用k近邻算法将每组数据划分到某个类中
其伪代码如下：
对未知类别属性的数据集中的每个点依次执行以下操作：
(1) 计算已知类别数据集中的点与当前点之间的距离；
(2) 按照距离递增次序排序；
(3) 选取与当前点距离最小的k个点；
(4) 确定前k个点所在类别的出现频率；
(5) 返回前k个点出现频率最高的类别作为当前点的预测分类。

参数：
inX：用于分类的输入向量
dataSet：输入的训练样本集，行数是数据的个数，列数是数据集特征值的个数
labels：标签向量，即分类的数目，标签向量的元素数目和矩阵dataSet的行数相同
k：用于选择最近邻居的数目
"""
def classify0(inX, dataSet, labels, k):
    #距离计算
    # shape返回矩阵的[行数，列数]，
    # 那么shape[0]获取数据集的行数，
    # 行数就是样本的数量
    dataSetSize = dataSet.shape[0]
    """
    计算输入向量和训练样本集之间的欧氏距离
    将inX重复dataSetSize行，然后减去dataSet中的每个元素，正好构成了
    [inX1-x00,inX2-x01,...,inXn-x0n,
     ...
     inX1-xm0,inX2-xm1,...,inXn-xmn]
     利用**每个元素平方
     axis=0表述列，axis=1表述行，即每行元素求和
     开根号，获得输入向量和训练样本集每个样本之间的欧氏距离
    """
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    #选择距离最小的k个点
    sortedDistIndicies = distances.argsort()  #argsort()方法得到矩阵中每个元素的排序序号，从小到大
    classCount = {}
    #统计k个点中各个分类的个数
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    #按照第二个元素的次序对元组进行排序，即各个分类的个数
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])    #4行表示4个输入点
    labels = ['A','A','B','B']   #4个点的分类
    return group, labels

"""
将文本记录转换为NumPy的解析程序
"""
def file2matrix(filename):
    #得到文件行数
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    fr = open(filename)
    index = 0

    #解析文件数据到列表
    for line in fr.readlines():
        #首先使用函数line.strip()截取掉所有的回车字符，然后使用tab字符\t将上一步得到的整行数据分割成一个元素列表
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        #必须明确地通知解释器，告诉它列表中存储的元素值为整型，否则Python语言会将这些元素当作字符串处理。
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    #returnMat：n行3列的样本数据，classLabelVector：样本数据的分类
    return returnMat,classLabelVector

"""
归一化特征值
dataSet：n行3列
由于某些特征的取值范围很大，导致这个特征对结果影响过大
在处理这种不同取值范围的特征值时，我们通常采用的方法是将数值归一化，如将取值范围处理为0到1或者-1到1之间。
下面的公式可以将任意取值范围的特征值转化为0到1区间内的值:
newValue  = (oldValue - min)/(max - min)
其中min和max分别是数据集中的最小特征值和最大特征值。
"""
def autoNorm(dataSet):
    #参数0使得函数可以从列中选取最小值，而不是选取当前行的最小值。注意minVals为1x3列的矩阵
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]#矩阵行数
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals

"""
测试上面的分类器
"""
def datingClassTest():
    hoRatio = 0.10    #选择总数据的10%进行测试
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')#加载数据
    normMat, ranges, minVals = autoNorm(datingDataMat)#归一化数据
    m = normMat.shape[0]  #样本数据的行数
    numTestVecs = int(m*hoRatio)  #测试数据的数目
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount

"""
手写识别系统,将图像转换为向量
该函数创建1×1024的NumPy数组，然后打开给定的文件，循环读出文件的前32行，并将每行的头32个字符值存储在NumPy数组
"""
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

"""
手写数字识别系统的测试代码
"""
def handwritingClassTest():
    hwLabels = []      #第i个文件的分类数字
    trainingFileList = listdir('digits/trainingDigits')
    #文件数目
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))    #每一行表示一个图像的样本点
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #这两步解析分类数字，注意文件名类似于0_0.txt
        classNumStr = int(fileStr.split('_')[0])  #要分类的数字
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('digits/testDigits')        #测试文件
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))