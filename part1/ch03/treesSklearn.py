'''
Decision Tree Source Code for Machine Learning in Action Ch. 3

@author: Teddy Ma
'''

# hack for loading shared module even when current folder is a sub folder
import sys
if (sys.path[-1] != '..'): sys.path.append('..')

from shared.common import *
from math import log
from numpy import array
from sklearn import tree
import pydotplus

def createDataSet():
    dataSet = array([[1, 1, 1],
               [1, 1, 1],
               [1, 0, 0],
               [0, 1, 0],
               [0, 1, 0]])
    labels = ['no surfacing','flippers']
    return dataSet, labels

def createTreeClf():
    myDat,labels=createDataSet()
    trainData = myDat[:,:2]
    labelIndexs = myDat[:,-1]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(trainData, labelIndexs)
    return clf,labels

# 3.3.1 classify test
def classifyTest():
    clf,labels = createTreeClf()
    return labels[int(clf.predict([[1,0]]))]

def plotTree():
    clf,labels = createTreeClf()
    dot_data = tree.export_graphviz(clf, out_file=None)    
    graph = pydotplus.graph_from_dot_data(dot_data)
    return createImage(graph)
