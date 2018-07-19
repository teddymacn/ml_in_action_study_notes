'''
Adaptive Boosting Source Code for chatpter 7

@author: Teddy.Ma
'''
import sys
if (sys.path[-1] != '..'): sys.path.append('..')

from shared.common import *
from numpy import *
from sklearn.ensemble import AdaBoostClassifier

def loadDataSet(fileName):
    data = loadTable(fileName)
    dataMat = data[:,:data.shape[1] - 1].tolist()
    labelMat = data[:,data.shape[1] - 1].tolist()
    return dataMat,labelMat

# 7.6 test ada boosting on horse colic dataset
def testAdaColic():
    datArr,labelArr = loadDataSet('horseColicTraining2.txt')
    clf = AdaBoostClassifier(n_estimators=100,algorithm='SAMME')
    clf.fit(datArr,labelArr)
    testArr,testLabelArr = loadDataSet('horseColicTest2.txt')
    prediction10 = mat(clf.predict(testArr).tolist()).transpose()
    errArr = mat(ones((len(testArr),1)))
    errRate = errArr[prediction10!=mat(testLabelArr).T].sum() / len(testArr)
    print ("erro rate:", errRate)
