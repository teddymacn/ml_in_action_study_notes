'''
Plotting Decision Trees Source Code for Machine Learning in Action Ch. 3

@author: Teddy Ma
'''

# hack for loading shared module even when current folder is a sub folder
import sys
if (sys.path[-1] != '..'): sys.path.append('..')

from shared.common import *
from trees import *

def plotTree(treeDic):
    g = createDiGraph()
    for root in treeDic.keys():
        createNode(root, 'box', g)
        for branch in treeDic[root].keys():
            plotBranch(g, treeDic[root][branch], root, branch)
    return createImage(g)
        
def plotBranch(graph, branchDic, parentName, branchName):
    if isinstance(branchDic, str):
        childName = branchDic
    else:
        for name in branchDic.keys():
            childName = name
            for branch in branchDic[childName].keys():
                plotBranch(graph, branchDic[childName][branch], parentName + '_' + childName, branch)  
    createNode(parentName + '_' + childName, 'box', graph,  childName)
    createEdge(parentName, parentName + '_' + childName, graph, branchName)

# 3.2.2 Figure 3.6 plot test tree
def plotTestTree():
    tree = createTestTree()
    return plotTree(tree)
    
# 3.2.2 Figure 3.7 plot test tree with maybe
def plotTestTree2():
    tree = createTestTree()
    tree['no surfacing'][3]='maybe'
    return plotTree(tree)
    
# 3.4 Figure 3.8 plot lenses tree
def plotLensesTree():
    tree = createLensesTree()
    return plotTree(tree)

