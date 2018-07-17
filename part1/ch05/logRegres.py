'''
Logistic Regression Source Code for chapter 05

@author: Teddy.Ma
'''
import sys
if (sys.path[-1] != '..'): sys.path.append('..')

from shared.common import *
from numpy import *

def loadDataSet():
    data = loadTable('testSet.txt')
    dataMat = empty((data.shape[0], 3))
    dataMat[:,0] = array(ones((data.shape[0], 1)))[:,0]
    dataMat[:,1:3] = data[:,:2]
    labelMat = data[:,2].astype('int')
    return dataMat.tolist(),list(labelMat)

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

# 5.2.2 simple gradient ascent
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
    dataTrain = loadTable('horseColicTraining.txt')
    trainingSet = dataTrain[:,:21]
    trainingLabels = dataTrain[:,21]
    trainWeights = stocGradAscent1(trainingSet, trainingLabels, 1000)
    
    errorCount = 0; numTestVec = 0.0
    dataTest = loadTable('horseColicTest.txt')
    for line in dataTest:
        numTestVec += 1.0
        if int(classifyVector(line[:21], trainWeights))!= int(line[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print ("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))
        