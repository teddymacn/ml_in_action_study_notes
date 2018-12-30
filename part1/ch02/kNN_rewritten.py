'''
kNN_rewritten: k Nearest Neighbors All-in-one Rewritten Version

@author: Teddy.Ma
'''

from numpy import *
import operator as op
import pandas as pd
from os import listdir, path
import matplotlib
import matplotlib.pyplot as plt
import time
import sys

currrentDir = path.dirname(path.realpath(__file__))

# return a sorted dictionary item list
def dicSorted(dicItems, desc = True):
    return sorted(dicItems, key=op.itemgetter(1), reverse=desc)

# return a sorted pair list
def pairSorted(pairItems, desc = True):
    return sorted(pairItems, key=lambda pair: pair[1], reverse=desc)

# load a numpy array from txt file
def loadTable(file):
    return array(pd.read_table(file,header=None))

# convert type of specified column, return the converted column of data
def convertColumnType(dataSet, columnIndex, targetType):
    mat = empty(dataSet.shape[0], dtype=targetType)
    mat[:] = dataSet[:,columnIndex]
    return mat

# 2.1.2 the core kNN algorithm
def classify0(testVect, mat, labels, k):
    m = mat.shape[0]
    
    # diff each vect from testVect
    diffMat = tile(testVect, (m,1)) - mat
    
    # calculate distances of testVect from each vect
    distances = ((diffMat**2).sum(axis=1))**0.5
    
    # sort by distance asc and get indices of mat
    sortedDistIndicies = distances.argsort()
    
    # calculate label votes of top k nearest mat
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    
    # return top voted label
    return dicSorted(classCount.items())[0][0]

# 2.1.2 create simple dataset for training & testing
def createDataSet():
    mat = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return mat, labels

# 2.1.2 simple class test with classify0() and createDataSet()
def simpleClassTest(vect):
    mat, labels = createDataSet()
    print("mat:", mat, sep="\n")
    print("labels:", labels, sep="\n")
    return classify0(vect, mat, labels, 3)

# 2.2.1 loading dating dataset1
def file2matrix():
    data = loadTable(currrentDir + '/datingTestSet.txt') # load raw data
    
    mat = data[:,0:3] # first 3 columns is data matrix
    
    # the last column is the labels
    labels = data[:,-1]
    
    return mat, labels
    
# 2.2.1 loading dating dataset2
def file2matrix2():
    data = loadTable(currrentDir + '/datingTestSet2.txt') # load raw data
    
    mat = data[:,0:3] # first 3 columns is data matrix
    
    # the last column is the label index, convert to int
    labels = convertColumnType(data, -1, int)
    
    return mat, labels

# 2.2.2 Figure 2.3 plot dating data simple
def plotDatingDataMat():
    datingDataMat,datingLabels = file2matrix2()
    fig = plt.figure("plotDatingDataMat")
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
    plt.show()

# 2.2.2 Figure 2.4 plot dating data with different marker size and color
def plotDatingDataMat2():
    datingDataMat,datingLabels = file2matrix2()
    fig = plt.figure("plotDatingDataMat2")
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 
       15.0*array(datingLabels), 15.0*array(datingLabels))
    plt.show()

# 2.2.2 Figure 2.5 plot dating data with frequent flier miles vs % of time 
# spent playing video game
def plotDatingDataMat3():
    datingDataMat,datingLabels = file2matrix2()
    fig = plt.figure("plotDatingDataMat3")
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,0], datingDataMat[:,1], 
       15.0*array(datingLabels), 15.0*array(datingLabels))
    plt.show()

# 2.2.3 generic auto nomalization 
def autoNorm(dataSet):
    minVals = dataSet.min(axis=0)
    maxVals = dataSet.max(axis=0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = (dataSet - tile(minVals, (m,1))) / tile(ranges, (m,1))
    return normDataSet, ranges, minVals

# 2.2.4 the first dating class test with string labels
def datingClassTest():
    hoRatio = 0.10      #hold out 10% as test set
    datingMat,datingLabels = file2matrix()
    normMat, ranges, minVals = autoNorm(datingMat)
    m = normMat.shape[0]
    numTestMat = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestMat):
        classifierResult = classify0(normMat[i,:],normMat[numTestMat:m,:], datingLabels[numTestMat:m],3)
        #print ("the classifier came back with: %s, the real answer is: %s" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
        
    print ("the total error rate is: %f" % (errorCount/float(numTestMat)))    

# 2.2.4 the second dating class test with int label index instead
def datingClassTest2():
    hoRatio = 0.50      #hold out 50% as test set
    datingMat,datingLabels = file2matrix2()       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingMat)
    m = normMat.shape[0]
    numTestMat = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestMat):
        classifierResult = classify0(normMat[i,:],normMat[numTestMat:m,:],datingLabels[numTestMat:m],3)
        #print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
        
    print ("the total error rate is: %f" % (errorCount/float(numTestMat)))

# 2.3.1 convert img to vector
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

# 2.3.2 testing kNN on handwritten digits
def handwritingClassTest():
    hwLabels = []
    print("loading training set...")
    trainingFileList = listdir(currrentDir + '/trainingDigits')    #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector(currrentDir + '/trainingDigits/%s' % fileNameStr)
    print("testing on test set: ")
    testFileList = listdir(currrentDir + '/testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(currrentDir + '/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        #print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
            print("x", end="")
        else:
            print(".", end="")
        sys.stdout.flush()
    print ("\nthe total number of errors is: %d out of %d" % (errorCount, mTest))
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))


if __name__ == "__main__":
    print("Simple dataset test\n")
    vect = [0,0]
    print("Classify result of vect %s: %s" % (vect, simpleClassTest(vect))) 
#    print("\n--\n")
#    print("Plotting dating dataset\n")
#    plt.ion()
#    plotDatingDataMat()
#    plotDatingDataMat2()
#    plotDatingDataMat3()
    print("\n--\n")
    print("Dating class test with string labels, hold out 10%\n")
    datingClassTest()
    print("\n--\n")
    print("Dating class test with label indices, hold out 50%\n")
    datingClassTest2()
    print("\n--\n")
    print("Handwriting test\n")
    handwritingClassTest()
