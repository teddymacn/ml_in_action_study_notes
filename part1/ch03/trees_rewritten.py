'''
trees_rewritten: Decision Trees All-in-one Rewritten Version

@author: Teddy Ma
'''

from math import log
import numpy as np
import operator as op
import pydotplus as dot
from IPython.display import Image, display
from PIL import Image as Image2
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd

currrentDir = os.path.dirname(os.path.realpath(__file__))

# return a sorted dictionary item list
def dicSorted(dicItems, desc = True):
    return sorted(dicItems, key=op.itemgetter(1), reverse=desc)

# load a numpy array from txt file
def loadTable(file):
    return np.array(pd.read_table(file,header=None))

# 3.1.1 create simple dataset for training & testing
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    featureLabels = ['no surfacing','flippers']
    return dataSet, featureLabels

# 3.1.1 calculate shannon entropy of a dataset
def calcShannonEnt(dataSet):
    # calculate the the number of unique elements and their occurance
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    
    # calculate shannon entropy
    shannonEnt = 0.0
    numEntries = len(dataSet)
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
        
    return shannonEnt

# 3.1.2 dataset splitting on a specified feature value
def splitDataSet(dataSet, featIndex, featValue):
    retDataSet = []
    for featVec in dataSet:
        if featVec[featIndex] == featValue:
            reducedFeatVec = featVec[:featIndex]     #chop out featIndex used for splitting
            reducedFeatVec.extend(featVec[featIndex+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 3.1.2 calculate info gain on each feature, return the best feature index
def chooseBestFeatureToSplit(dataSet):
    numFeats = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEnt = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeatIndex = -1
    for i in range(numFeats):        #iterate over all the features
        featList = [example[i] for example in dataSet] #create a list of all the examples of this feature   
        newEnt = 0.0
        for value in set(featList): #for each unique feature value
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEnt += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEnt - newEnt     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeatIndex = i
            
    return bestFeatIndex                      #returns the best feature index

# 3.1.2 return the top voted label
def majorityCnt(labels):
    labelCount={}
    for vote in labels:
        labelCount[vote] = labelCount.get(vote, 0) + 1
    sortedLabelCount = dicSorted(labelCount.items())
    return sortedLabelCount[0][0]

# 3.1.3 recursively create a tree
def createTree(dataSet,featureLabels):
    parentFeatLabels = featureLabels[:] # clone the featureLabels to avoid changing the source list

    classLabels = [example[-1] for example in dataSet]
    if classLabels.count(classLabels[0]) == len(classLabels): 
        return classLabels[0] #stop splitting when all of the classes are equal
    
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classLabels)
    
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = parentFeatLabels[bestFeat]
    
    myTree = {bestFeatLabel:{}}
    del(parentFeatLabels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    for value in set(featValues): # for each unique feature value
        subFeatLabels = parentFeatLabels[:]       #copy all of featureLabels, so trees don't mess up existing featureLabels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subFeatLabels)

    return myTree                            

# create a digraph
def createDiGraph():
    g = dot.Dot();
    g.set_type('digraph')
    return g

# create graph node
def createNode(name, shape=None, graph=None, label=None, style=None, fillcolor=None):
    node = dot.Node(name)
    if shape is not None: node.set("shape", shape)
    if graph is not None: graph.add_node(node)
    if label is not None: node.set('label', label)
    if style is not None: node.set('style', style)
    if fillcolor is not None: node.set('fillcolor', fillcolor)
    return node

# create a graph edge
def createEdge(src, dst, graph=None, label=None):
    edge = dot.Edge(src, dst)
    if graph is not None: graph.add_edge(edge)
    if label is not None: edge.set('label', label)
    return edge

# show graph as IPython image
def showIPythonImage(graph):
    display(Image(graph.create_png()))

# show graph as image file
def showTempFileImage(graph):
    file = "temp_plot_tree.png"
    graph.write_png(file)
    img = Image2.open(file)
    img.show()
    os.remove(file)

# 3.2 plot a tree with pydotplus
def plotTree(treeDic):
    g = createDiGraph()
    for root in treeDic.keys():
        createNode(root, 'diamond', g)
        for branch in treeDic[root].keys():
            plotBranch(g, treeDic[root][branch], root, branch)
    try:
        __IPYTHON__
        showIPythonImage(g)
    except NameError:
        showTempFileImage(g)

# 3.2 plot a tree branch with pydotplus
def plotBranch(graph, branchDic, parentName, branchName):
    nodeStyle = None
    nodeColor = None
    nodeShape = 'diamond'
    if isinstance(branchDic, str):
        childName = branchDic
        nodeStyle = 'filled'
        nodeColor = 'lightgray'
        nodeShape = 'box'
    else:
        for name in branchDic.keys():
            childName = name
            for branch in branchDic[childName].keys():
                plotBranch(graph, branchDic[childName][branch], parentName + '_' + childName, branch)  
    createNode(parentName + '_' + childName, nodeShape, graph,  childName, style=nodeStyle, fillcolor=nodeColor)
    createEdge(parentName, parentName + '_' + childName, graph, branchName)

# 3.3.1 classify with decision tree
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
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)

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
    lensesDataSet=loadTable(currrentDir + '/lenses.txt').tolist()
    lensesLabels=['age', 'prescript', 'astigmatic', 'tearRate']
    return createTree(lensesDataSet,lensesLabels)
    
if __name__ == "__main__":
    print("Create a tree for the given dataset & feature labels\n")
    dataset, featureLabels = createDataSet()
    print("dataset: ", dataset)
    print("feature labels: ", featureLabels)
    myTree = createTree(dataset, featureLabels)
    print("\nthe created tree: ", myTree, sep="\n")
    plotTree(myTree)
    print("Classify with decision tree:")
    print([1, 0], " -> ", classify(myTree,featureLabels,[1,0]), sep="")
    print([1, 1], " -> ", classify(myTree,featureLabels,[1,1]), sep="")
    print("\n---\n")
    print("Store & grab tree from file:")
    print(storeGrabTreeTest())
    print("\n---\n")
    print("Create the lenses tree:")
    lensesTree = createLensesTree()
    print(lensesTree)
    plotTree(lensesTree)
