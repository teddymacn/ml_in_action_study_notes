'''
Plotting Logistic Regression Source Code for Machine Learning in Action Ch. 5

@author: Teddy Ma
'''

from logRegres import *

import matplotlib
import matplotlib.pyplot as plt
from numpy import *

# 5.1 plot sigmoid functions
def plotSigmoid():
    x = linspace(-5, 5, 1000)
    plt.plot(x, sigmoid(x))
    plt.show()
    x = linspace(-60, 60, 1000)
    plt.plot(x, sigmoid(x))
    plt.show()

# 5.2 plot a best-fit linear equation function
def plotBestFit(weights):
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

# 5.2.3 plot BestFit for GradAscent()
def plotBestFitGradAscent():
    dataArr,labelMat = loadDataSet()
    weights = gradAscent(dataArr,labelMat)
    print (weights)
    plotBestFit(weights)
    

# 5.2.4 plot for StocGradAscent0
def plotBestFitStocGradAscent0():
    dataArr,labelMat = loadDataSet()
    weights = stocGradAscent0(array(dataArr),labelMat)
    print (weights)
    plotBestFit(weights)

# 5.2.4 plot for StocGradAscent1
def plotBestFitStocGradAscent1():
    dataArr,labelMat = loadDataSet()
    weights = stocGradAscent1(array(dataArr),labelMat)
    print (weights)
    plotBestFit(weights)
    
