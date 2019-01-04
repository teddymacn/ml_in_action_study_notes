'''
logRegres_rewritten: Logistic Regression All-in-one Rewritten Version

@author: Teddy.Ma
'''

from numpy import *
import operator as op
import os
import pandas as pd
import matplotlib.pyplot as plt

currrentDir = os.path.dirname(os.path.realpath(__file__))

# load a numpy array from txt file
def loadTable(file):
    return array(pd.read_table(file,header=None))

# 5.2 plot a best-fit linear equation function
def plotBestFit(weights):
    dataMat,labelMat=loadDataSet()
    m = shape(dataMat)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(m):
        if int(labelMat[i])== 1:
            xcord1.append(dataMat[i,1]); ycord1.append(dataMat[i,2])
        else:
            xcord2.append(dataMat[i,1]); ycord2.append(dataMat[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

# 5.2.2 Simple test dataset
def loadDataSet():
    data = loadTable(currrentDir + '/testSet.txt')
    dataMat = empty((data.shape[0], 3))
    dataMat[:,0] = array(ones((data.shape[0], 1)))[:,0]
    dataMat[:,1:3] = data[:,:2]
    labelMat = data[:,2].astype('int')
    return dataMat, labelMat

# 5.2.2 The sigmoid function
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

# 5.2.2 simple gradient ascent algorithm
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult
        error = (labelMat - h)              #vector subtraction
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    return array(mat(weights).transpose())[0]

# 5.2.4 stochastic gradient ascent
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

# 5.2.4 modified stochastic gradient ascent
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not 
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

# 5.3.2 test logistic Regression
def colicTest():
    dataTrain = loadTable(currrentDir + '/horseColicTraining.txt')
    trainingSet = dataTrain[:,:21]
    trainingLabels = dataTrain[:,21]
    trainWeights = stocGradAscent1(trainingSet, trainingLabels, 100)
    
    errorCount = 0; numTestVec = 0.0
    dataTest = loadTable(currrentDir + '/horseColicTest.txt')
    for line in dataTest:
        numTestVec += 1.0
        if int(classifyVector(line[:21], trainWeights))!= int(line[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print ("the error rate of this test is: %f" % errorRate)
    return errorRate

if __name__ == "__main__":
    dataMat, labels = loadDataSet()
    print("Gradient ascent:")
    w1 = gradAscent(dataMat, labels)
    print(w1)
    #plotBestFit(w1)
    print("\n---\n")
    print("Simple stochatic gradient ascent does not work very well:")
    w2 = stocGradAscent0(dataMat, labels)
    print(w2)
    #plotBestFit(w2)
    print("\n---\n")
    print("Modified stochatic gradient ascent, alpha changes with each iteration, works much better:")
    w3 = stocGradAscent1(dataMat, labels)
    print(w3)
    #plotBestFit(w3)
    print("\n---\n")
    print("Colic Test:")
    colicTest()
    
