#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
朴素贝叶斯

@Author: xie
@Date: 2017/12/6
"""
from numpy import *

"""
词表到向量的转换函数，该函数创建了一些实验样本。
postingList：是进行词条切分后的文档集合，这些文档来自斑点犬爱好者留言板。这些留言文本被切分成一系列的词条集合，标点符号从文本中去掉。
classVec：是一个类别标签的集合。这里有两类，侮辱性和非侮辱性。这些文本的类别由人工标注，这些标注信息用于训练程序以便自动检测侮辱性留言。

朴素贝叶斯分类器通常有两种实现方式:
一种基于贝努利模型实现，该实现方式中并不考虑词在文档中出现的次数，只考虑出不出现，因此在这个意义上相当于假设词是等权重的。这可以被描述为词集模型(set-of-words model)。
一种基于多项式模型实现，考虑词在文档中出现的次数，这种方法被称为词袋模型(bag-of-words model)。
"""
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]     #1 代表侮辱性文字，0代表正常言论，代表上面6个（每行一个文档）的分类
    return postingList,classVec

"""
创建一个包含在所有文档中出现的不重复词的列表
"""
def createVocabList(dataSet):
    vocabSet = set([])  # 创建一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document) #创建两个集合的并集，|也是一个按位或(OR)操作符
    return list(vocabSet)

"""
输入参数为词汇表及某个文档，输出的是文档向量，向量的每一元素为1或0，分别表示词汇表中的单词在输入文档中是否出现。
这里基于贝努利模型实现。
函数首先创建一个和词汇表等长的向量，并将其元素都设置为0 。
接着，遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1。
"""
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)   #创建一个其中所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return returnVec

"""
朴素贝叶斯分类器训练函数
trainMatrix:文档矩阵,矩阵的行数为文档的数目,
            矩阵的列数为文档向量，向量的每一元素为1或0，分别表示词汇表中的单词在输入文档中是否出现
trainCategory:由每篇文档类别标签所构成的向量
返回两个向量和一个概率，即：p(w_i|c_0)向量，p(w_i|c_1)向量，p(c_1)文档属于侮辱性文档(class=1)的概率

该函数的伪代码如下:
  计算每个类别中的文档数目
  对每篇训练文档:
    对每个类别:
      如果词条出现在文档中→ 增加该词条的计数值 
      增加所有词条的计数值
    对每个类别:
      对每个词条:
        将该词条的数目除以总词条数目得到条件概率
    返回每个类别的条件概率
"""
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)    #矩阵的行数为文档的数目
    numWords = len(trainMatrix[0])     #矩阵的列数为不重复词的数量，大小等于词汇表大小
    pAbusive = sum(trainCategory)/float(numTrainDocs)   #计算文档属于侮辱性文档(class=1)的概率，即P(1)
    """
    利用贝叶斯分类器对文档进行分类时，要计算多个概率的乘积以获得文档属于某个类别的概率，即计算p(w0|1)p(w1|1)p(w2|1)。
    如果其中一个概率值为0，那么最后的乘积也为0。为降低这种影响，可以将所有词的出现数初始化为1，并将分母初始化为2。
    这是由于我们假设所有的词条出现相互独立，使计算条件概率时分子或者分母为零。
    """
    p0Num = ones(numWords); p1Num = ones(numWords)      #初始化概率
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):     #遍历文档
        if trainCategory[i] == 1:   #侮辱性文档
            p1Num += trainMatrix[i]    #侮辱性文档每个词条出现的频率
            p1Denom += sum(trainMatrix[i])   #侮辱性文档词条出现总数
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    """
    另一个遇到的问题是下溢出，这是由于太多很小的数相乘造成的。
    当计算乘积 p(w0|ci)p(w1|ci)p(w2|ci)...p(wN|ci)时，由于大部分因子都非常小，所以程序会下溢出或者得到不正确的答案。
    (读者可以用Python尝试相乘许多很小的数，最后四舍五入后会得到0。)
    一种解决办法是对乘积取自然对数。在代数中有ln(a*b) = ln(a)+ln(b)，于是通过求对数可以避免下溢出或者浮点数舍入导致的错误。
    同时，采用自然对数进行处理不会有任何损失。
    """
    p1Vect = log(p1Num/p1Denom)         #侮辱性文档每个词条出现的概率，即p(w_i|c_1)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

"""
朴素贝叶斯分类函数
vec2Classify：要分类的向量
其他：使用函数trainNB0()计算得到的三个概率。
"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #元素相乘，即p1=log p(c_1)*p(w_1|c_1)*p(w_2|c_1)*,...,*p(w_N|c_1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)   #p0=log p(c_0)*p(w_1|c_0)*p(w_2|c_0)*,...,*p(w_N|c_0)
    if p1 > p0:    #即p1/{p(w_1)*p(w_2)*,...,*p(w_N)} > p0/{p(w_1)*p(w_2)*,...,*p(w_N)}，消去分母不影响最终结果
        return 1
    else: 
        return 0

"""
这里基于多项式模型实现
"""
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

"""
便利函数(convenience function)，该函数封装所有操作，以节省输入上面代码的时间
"""
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))

    # 测试文档
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)

"""
正则表达式分词
输入String文本，输出单词列表
"""
def textParse(bigString):
    import re
    """
    Python中字符串前面加上r表示原生字符串，否则要转义\，即：'\\W*'。
    与大多数编程语言相同，正则表达式里使用"\"作为转义字符，这就可能造成反斜杠困扰。
    假如你需要匹配文本中的字符"\"，那么使用编程语言表示的正则表达式里将需要4个反斜杠"\\\\"：前两个和后两个分别用于在编程语言里转义成反斜杠，转换成两个反斜杠后再在正则表达式里转义成一个反斜杠。
    Python里的原生字符串很好地解决了这个问题，这个例子中的正则表达式可以使用r"\\"表示。同样，匹配一个数字的"\\d"可以写成r"\d"。有了原生字符串，你再也不用担心是不是漏写了反斜杠，写出来的表达式也更直观。
    """
    listOfTokens = re.split(r'\W*', bigString)   #只匹配任意多个字母、数字、下划线的字符
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] #找出长度大于2的单词，并且全部转换为小写

"""
贝叶斯垃圾邮件分类器进行自动化处理
"""
def spamTest():
    docList=[]; classList = []; fullText =[]

    # 导入并解析文本文件
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)  #换行添加list
        fullText.extend(wordList)  #添加到同一个list中
        classList.append(0)
    vocabList = createVocabList(docList)  #创建一个包含在所有文档中出现的不重复词的列表

    """
    随机构建训练集，这种随机选择数据的一部分作为训练集，而剩余部分作为测试集的过程称为留存交叉验证(hold-out cross validation)。
    假定现在只完成了一次迭代，那么为了更精确地估计分类器的错误率，就应该进行多次迭代后求出平均错误率。
    """
    trainingSet = range(50); testSet=[]
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))  #随机生成0-len(trainingSet)的int值
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  #为了不重复获得值，删除掉已经获得的值，下次循环这个值就不在trainingSet中，并且len(trainingSet)-1
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))

    #对测试集分类
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error",docList[docIndex]
    print 'the error rate is: ',float(errorCount)/len(testSet)
    #return vocabList,fullText

"""
RSS源分类器及高频词去除函数
该函数遍历词汇表中的每个词并统计它在文本中出现的次数，然后根据出现次数从高到低对词典进行排序，最后返回排序最高的30个单词。
vocabList：一个包含在所有文档中出现的不重复词的列表
fullText：将所有的文档解析后放到一个大的list中，允许重复
"""
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) 
    return sortedFreq[:30]       

"""
使用两个RSS源作为参数。
与spamTest()函数几乎相同，区别在于这里访问的是RSS源而不是文件，然后排序最高的30个单词并将他们移除
（注意移除只是为了测试，移除后错误率甚至会上升，实际上可以将一些白名单词放进去）
"""
def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))#文章数的较小值
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)   #创建一个包含在所有文档中出现的不重复词的列表
    top30Words = calcMostFreq(vocabList,fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])

    #随机构建训练集
    trainingSet = range(2*minLen); testSet=[]
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))

    #剩下的用来测试
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ',float(errorCount)/len(testSet)
    return vocabList,p0V,p1V

"""
选择概率大于某个阈值的所有词
"""
def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))

    #按照单词出现的条件概率排序，而不是calcMostFreq()中的词频
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0]
