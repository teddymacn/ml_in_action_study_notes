'''
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: Teddy.Ma
'''
# hack for loading shared module even when current folder is a sub folder
import sys
if (sys.path[-1] != '..'): sys.path.append('..')

from shared.common import *
from numpy import *
from sklearn import neighbors, preprocessing
from os import listdir

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

# 2.1.2 simple class test with classify0() and createDataSet()
def simpleClassTest():
    group, labels = createDataSet()
    clf = neighbors.KNeighborsClassifier(3)
    clf.fit(group, labels)
    testX = array([0,0]).reshape(1,-1)
    return clf.predict(testX);

def file2matrix():
    data = loadTable('datingTestSet.txt') # load raw data
    
    returnMat = data[:,0:3] # first 3 columns is data matrix
    
    # the last column is the labels
    datingLabels = data[:,-1]
    
    return returnMat,datingLabels
    
def file2matrix2():
    data = loadTable('datingTestSet2.txt') # load raw data
    
    returnMat = data[:,0:3] # first 3 columns is data matrix
    
    # the last column is the label index, convert to int
    classLabelVector = empty(data.shape[0], dtype=int)
    classLabelVector[:] = data[:,-1]
    
    return returnMat,classLabelVector
    
def autoNorm(dataSet):
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(dataSet)
    normDataSet = scaler.transform(dataSet)
    return normDataSet, scaler.data_range_ , scaler.data_min_
   
# 2.2.1 the first dating class test with string labels
def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix()
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    clf = neighbors.KNeighborsClassifier(3)
    clf.fit(normMat[numTestVecs:m,:], datingLabels[numTestVecs:m])
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = clf.predict(normMat[i,:].reshape(1,-1))[0]
        print ("the classifier came back with: %s, the real answer is: %s" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
        
    print ("the total error rate is: %f" % (errorCount/float(numTestVecs)))    

# 2.2.4 the second dating class test with int label index instead
def datingClassTest2():
    hoRatio = 0.50      #hold out 50%
    datingDataMat,datingLabels = file2matrix2()       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    clf = neighbors.KNeighborsClassifier(3)
    clf.fit(normMat[numTestVecs:m,:], datingLabels[numTestVecs:m])    
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = clf.predict(normMat[i,:].reshape(1,-1))[0]
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print ("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print (errorCount)
    
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    clf = neighbors.KNeighborsClassifier(3)
    clf.fit(trainingMat, hwLabels)        
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = clf.predict(vectorUnderTest.reshape(1,-1))[0]
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))