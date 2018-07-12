'''
Decision Tree Source Code for Machine Learning in Action Ch. 3

@author: Teddy Ma
'''

# hack for loading shared module even when current folder is a sub folder
import sys
if (sys.path[-1] != '..'): sys.path.append('..')

from shared.common import *
from math import log

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt
    
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
    
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = dicSorted(classCount.items())
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    parentLabels = labels[:] # clone the labels to avoid changing the source list
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = parentLabels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(parentLabels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = parentLabels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            
    
def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)
    
# 3.1.1 calculate entropy test
def calcShannonEntTest():
    myDat,labels=createDataSet()
    return calcShannonEnt(myDat)

# 3.1.2 split dataset test
def splitDataSetTest(value=0):
    myDat,labels=createDataSet()
    return splitDataSet(myDat,0,value)

# 3.1.2 choose best feature to split test
def chooseBestFeatureToSplitTest():
    myDat,labels=createDataSet()
    return chooseBestFeatureToSplit(myDat)

# 3.1.3 create tree test
def createTreeTest():
    myDat,labels=createDataSet()
    return createTree(myDat,labels)

# 3.3.1 classify test
def classifyTest():
    myDat,labels=createDataSet()
    myTree = createTree(myDat,labels)
    return classify(myTree,labels,[1,0]), classify(myTree,labels,[1,1])

# 3.3.2 persist the tree test
def storeGrabTreeTest():
    myDat,labels=createDataSet()
    myTree = createTree(myDat,labels)
    storeTree(myTree, 'classifierStorage.txt')
    return grabTree('classifierStorage.txt')

# 3.4 create lenses tree
def createLensesTree():
    fr=open('lenses.txt')
    lenses=[inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels=['age', 'prescript', 'astigmatic', 'tearRate']
    return createTree(lenses,lensesLabels)
    
    