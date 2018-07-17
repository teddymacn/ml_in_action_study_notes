'''
Chapter 6 source file for Machine Learing in Action

@author: Teddy.Ma
'''
import sys
if (sys.path[-1] != '..'): sys.path.append('..')

from shared.common import *
from numpy import *
from sklearn.svm import SVC

# 6.3.2 load dataset from txt file
def loadDataSet(fileName):
    data = loadTable(fileName)
    dataMat = data[:,:2].tolist()
    labelMat = list(data[:,2])
    return dataMat,labelMat

# 6.5.3 Radial bias test
def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    clf = SVC()
    clf.fit(array(dataArr), array(labelArr))
    errorCount = 0
    for i in range(len(labelArr)):
        predict=clf.predict([dataArr[i]])[0]
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/len(dataArr)))
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    for i in range(len(labelArr)):
        predict=clf.predict([dataArr[i]])[0]
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print("the test error rate is: %f" % (float(errorCount)/len(dataArr)))
    
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels    

# 6.6 revisit handwriting classification
def testDigits(kTup=('rbf', 10)):
    dataArr,labelArr = loadImages('trainingDigits')
    clf = SVC()
    clf.fit(array(dataArr), array(labelArr))
    errorCount = 0
    for i in range(len(labelArr)):
        predict=clf.predict([dataArr[i]])[0]
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/len(dataArr)))
    dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    for i in range(m):
    for i in range(len(labelArr)):
        predict=clf.predict([dataArr[i]])[0]
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount)/len(dataArr)))
