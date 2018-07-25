'''

pca Source Code for chatpter 13

@author: Teddy.Ma

'''
import sys
if (sys.path[-1] != '..'): sys.path.append('..')

from shared.common import *
from numpy import *

def loadDataSet(fileName):
    datArr = loadTable(fileName)
    return mat(datArr)

# 13.2.2 The PCA algorithm
def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

# 13.2.2 test pca on testSet.txt
def testTestSetPca():
    dataMat = loadDataSet('testSet.txt')
    lowDMat, reconMat = pca(dataMat, 1)
    return lowDMat

# 13.2.2 test pca on testSet3.txt
def testTestSet3Pca():
    dataMat = loadDataSet('testSet3.txt')
    lowDMat, reconMat = pca(dataMat, 2)
    return lowDMat
