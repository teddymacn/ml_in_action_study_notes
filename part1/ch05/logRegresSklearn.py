'''
Logistic Regression Source Code for chapter 05

@author: Teddy.Ma
'''
import sys
if (sys.path[-1] != '..'): sys.path.append('..')

from shared.common import *
from numpy import *
from sklearn.linear_model import LogisticRegression

# 5.3.2 test logistic Regression
def colicTest():
    dataTrain = loadTable('horseColicTraining.txt')
    trainingSet = dataTrain[:,:21]
    trainingLabels = dataTrain[:,21]
    
    clf = LogisticRegression()
    clf.fit(array(trainingSet), array(trainingLabels))    
    errorCount = 0; numTestVec = 0.0
    dataTest = loadTable('horseColicTest.txt')
    for line in dataTest:
        numTestVec += 1.0
        if int(clf.predict([line[:21]]))!= int(line[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print ("the error rate of this test is: %f" % errorRate)
    return errorRate
