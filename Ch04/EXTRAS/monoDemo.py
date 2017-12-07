#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
测试log函数不改变函数凹凸特性

@Author: xiezhenwei
@Date: 2017/12/6
"""
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

t = arange(0.01, 0.5, 0.01)   #0.0到0.5，步长为0.01
s = sin(2*pi*t)
logS = log(s)

fig = plt.figure()
ax = fig.add_subplot(211)     #参数211的意思是：将画布分割成2行1列，图像画在从左到右从上到下的第1块
ax.plot(t,s)
ax.set_ylabel('f(x)')
ax.set_xlabel('x')

ax = fig.add_subplot(212)
ax.plot(t,logS)
ax.set_ylabel('ln(f(x))')
ax.set_xlabel('x')
plt.show()